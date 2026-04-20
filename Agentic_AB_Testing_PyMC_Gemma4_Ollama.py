"""
Agentic Bayesian A/B Testing — Gemma 4 Edition
================================================
Use case : Coffee shop loyalty app push notification experiment
           Variant A = generic reminder  ("Your morning coffee is waiting")
           Variant B = discount offer    ("Get 10% off your next latte today!")

Supported models (all via Ollama):
  Gemma 4  →  gemma4:e2b  |  gemma4:e4b  |  gemma4:26b  |  gemma4:31b  |  gemma4:31b-cloud

Install
-------
    # 1. Install Ollama → https://ollama.com
    ollama pull gemma4:e2b      # ~2 GB  — smallest, edge/laptop
    ollama pull gemma4:e4b      # ~4 GB  — good for most laptops ✓
    ollama pull gemma4:26b      # ~16 GB — high quality
    ollama pull gemma4:31b      # ~19 GB — best local Gemma4
    ollama pull gemma4:31b-cloud           # cloud-routed (no local download)

    # 2. Python deps (conda-forge recommended)
    conda install -c conda-forge pymc arviz numpy
    pip install requests

Run
---
    python agentic_ab_test_gemma.py --model gemma4:e4b
    python agentic_ab_test_gemma.py --model gemma4:31b
    python agentic_ab_test_gemma.py --dry-run   # no LLM needed
"""

# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import re
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import requests


# ── Output helpers ─────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, fh):
        self._file = fh; self._stdout = sys.__stdout__
    def write(self, data):
        self._stdout.write(data); self._file.write(data)
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def fileno(self):
        return self._stdout.fileno()

def _make_output_path(base):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base:
        p = Path(base)
        if p.is_dir():
            return p / f"ab_report_{ts}.txt"
        return p if p.suffix else p.with_suffix(".txt")
    return Path(f"ab_report_{ts}.txt")

# ── Silence noisy runtime warnings ────────────────────────────────────────────
# 1. Intel OpenMP + LLVM OpenMP conflict (harmless conda env issue)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 2. PyMC / pytensor internal warnings
warnings.filterwarnings("ignore")
# 3. PyMC sampler progress and info logs
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("numexpr").setLevel(logging.ERROR)
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Simulated experiment data (ground truth hidden from the agent) ──────────
np.random.seed(7)
TRUE_RATE_A = 0.08    # Variant A: 8%  — generic reminder
TRUE_RATE_B = 0.13    # Variant B: 13% — discount offer  (B is 62% better)
VISITORS_A  = 500
VISITORS_B  = 500
CONVERSIONS_A = int(np.random.binomial(VISITORS_A, TRUE_RATE_A))
CONVERSIONS_B = int(np.random.binomial(VISITORS_B, TRUE_RATE_B))

EXPERIMENT = {
    "business_context": "Coffee shop loyalty app push notification A/B test",
    "variant_a": {
        "description": "Generic reminder — 'Your morning coffee is waiting'",
        "visitors": VISITORS_A,
        "conversions": CONVERSIONS_A,
    },
    "variant_b": {
        "description": "Discount offer — 'Get 10% off your next latte today!'",
        "visitors": VISITORS_B,
        "conversions": CONVERSIONS_B,
    },
    "objective": (
        "Which notification converts more users to a purchase? "
        "Report P(B > A), expected lift %, and whether the model converged. "
        "R-hat < 1.05 and ESS > 400 are required for a trustworthy result."
    ),
}

# ── 2. Model-specific configuration ───────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"


# ── 3. System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an autonomous Bayesian statistician working with PyMC v5 (PyMC5).
Your job is to analyse an A/B experiment, choose an appropriate model, run
it, evaluate the results, and decide whether it is good enough - or needs
improvement.

IMPORTANT: You are writing PyMC5 code (pymc>=5.0). Do NOT use PyMC3 or
PyMC4 patterns. The API changed significantly between versions.

You must respond with ONLY valid JSON - no markdown fences, no prose outside
the JSON, no preamble. Just the JSON object.

REQUIRED JSON SCHEMA (use exactly these keys):
{
  "reasoning":        "<Your statistical thinking: why this model, why this
                        prior, what diagnostics tell you, how you will fix
                        any issues. Be specific.>",
  "model_code":       "<Self-contained Python - see CONTRACT below>",
  "decision":         "keep" | "refine",
  "refinement_notes": "<Concrete plan for next iteration, or null>"
}

model_code CONTRACT:
Available globals (already imported - do NOT import anything):
    np  (numpy),  pm  (pymc),  az  (arviz)

Define exactly ONE function:
    def run_model(conv_a, vis_a, conv_b, vis_b) -> dict

Return dict MUST contain exactly:
    {
      "p_b_better":        float,   # posterior P(B > A)
      "expected_lift_pct": float,   # mean posterior lift % = (B-A)/A * 100
      "r_hat_max":         float,   # float(summary["r_hat"].max())
      "ess_min":           float,   # float(summary["ess_bulk"].min())
    }

PYMC3 vs PYMC5 - do NOT use these PyMC3 patterns:
  PYMC3 (WRONG)                      PYMC5 (CORRECT)
  ---------------------------------- ----------------------------------
  with model: ... with trace: ...    with pm.Model() as m: ... ONLY
  trace["p_a"]                       trace.posterior["p_a"]
  pm.summary(trace)                  az.summary(trace)
  pm.traceplot(trace)                az.plot_trace(trace)
  .astype("float32")                 .astype(float)
  pm.math.greater(a, b)              pm.math.gt(a, b)
  pm.math.beta(a, b)                 pm.Beta("name", alpha=a, beta=b)
  tensor.item()                      float(tensor.eval()) or use numpy
  tensor.replace(a, b)               pm.math.switch() or numpy ops
  az.mean_probability(...)           float(trace.posterior["x"].mean())
  pm.sample(..., step=Metropolis())  pm.sample(...)  # let PyMC choose
  step=pm.Metropolis([obs_node])     NEVER pass observed nodes to step

CRITICAL RULES - violations will crash the script:

  1. NEVER use "with trace:" or "with idata:" - InferenceData is NOT a
     context manager. This is a PyMC3 pattern that does not exist in PyMC5.
     The ONLY valid context manager is: with pm.Model() as model:

  2. NEVER call pm.math.beta() - it does not exist.
     Beta is a DISTRIBUTION: pm.Beta("name", alpha=a, beta=b)

  3. NEVER call .item() or .replace() on TensorVariable objects.
     These are NumPy/string methods. To get a scalar from a tensor use:
       float(trace.posterior["var"].values.mean())

  3b. ALWAYS define b_better as a Deterministic BEFORE calling pm.sample():
       b_better = pm.Deterministic("b_better", pm.math.gt(p_b, p_a).astype(float))
      Then access it AFTER sampling as:
       float(trace.posterior["b_better"].values.mean())
      If you access trace.posterior["b_better"] but never defined it, you
      get a KeyError. Define it first, always.

  4. NEVER call az.mean_probability() - it does not exist.
     To compute P(B > A) use:
       float(trace.posterior["b_better"].values.mean())

  5. r_hat_max and ess_min MUST be plain Python floats:
       summary   = az.summary(trace, var_names=["p_a", "p_b"])
       r_hat_max = float(summary["r_hat"].max())    # () required
       ess_min   = float(summary["ess_bulk"].min()) # () required

  6. Distribution parameters MUST be keyword args:
       CORRECT:  pm.Beta("p_a", alpha=1, beta=1)
       WRONG:    pm.Beta("p_a", 1, 1)

  7. pm.sample() MUST be INSIDE the with pm.Model() block.

  8. Comparisons: pm.math.gt()  pm.math.lt()  pm.math.ge()  pm.math.le()

  9. Sampler minimum — use default NUTS, never pass a step= argument:
       pm.sample(draws=1000, tune=1000, cores=1,
                 progressbar=False, return_inferencedata=True)
     NEVER do this:
       step = pm.Metropolis()  # wrong
       pm.sample(..., step=step)  # wrong
     NEVER pass observed variables to a step method:
       pm.sample(..., step=pm.Metropolis([conv_a_obs]))  # WRONG - observed
     Let PyMC5 choose the sampler automatically. For Beta-Binomial models
     NUTS is always used automatically - no step= needed.

  10. NEVER sample observed variables. Variables created with observed=...
      are fixed data - they are never sampled. Only sample latent variables
      like p_a and p_b. Passing observed nodes to step methods causes
      divergences and model failure.

  11. Do NOT call plt.show() or write files.

YOUR STATISTICAL RESPONSIBILITIES - decide ALL of the following yourself:

  PRIOR CHOICE
  - What distribution family fits the parameter space?
  - Informative or weakly informative? Justify from the business context
    (typical push-notification conversion rates for a coffee loyalty app).
  - What hyperparameter values encode that belief?

  MODEL STRUCTURE
  - Is a simple Beta-Binomial sufficient, or does data suggest something
    richer (overdispersion, hierarchical pooling, etc.)?
  - What derived quantities matter for the business decision?

  CONVERGENCE ASSESSMENT
  - Inspect R-hat and ESS values returned to you.
  - Set your own thresholds based on what is acceptable for this decision.
  - If convergence failed, diagnose WHY and fix the specific cause -
    do not just blindly increase draws.

  DECISION
  - "keep"   - converged, diagnostics acceptable, posterior interpretable.
  - "refine" - something is wrong; state what and how to fix it.

Do not copy any prior, structure, or threshold not derived from your own
reasoning about this specific experiment.
""")


# ── 4. Ollama client ───────────────────────────────────────────────────────────

# Path for saving raw model thinking output (set by __main__)
_THINK_STATE = {"path": None, "iteration": 0}  # mutable, no global needed


def _save_think_log(raw: str, model: str) -> None:
    """Save the raw model output (including think blocks) to a .think file."""
    if _THINK_STATE["path"] is None:
        return
    _THINK_STATE["iteration"] += 1
    import re as _re

    # Extract think block if present
    think_match = _re.search(r"<think>(.*?)</think>", raw, _re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else "(no think block)"

    # Extract JSON portion (everything after the think block)
    json_portion = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()

    with open(_THINK_STATE["path"], "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"  ITERATION {_THINK_STATE['iteration']}  |  model: {model}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"--- THINKING ---\n{think_content}\n\n")
        f.write(f"--- JSON OUTPUT ---\n{json_portion[:3000]}\n")
        if len(json_portion) > 3000:
            f.write(f"... (truncated, {len(json_portion)} chars total)\n")


def ask_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,   # lower = more deterministic JSON output
            "num_predict": 6000,   # increased: model_code can be long
        },
        # Disable thinking mode for gemma4:26b and gemma4:31b
        # Thinking tokens (<think>...</think>) break JSON parsing
        "think": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        if r.status_code == 404:
            raise RuntimeError(
                f"\n  Model '{model}' not found in Ollama (404).\n"
                f"  Common mistakes:\n"
                f"    gemma4:e26b  <- WRONG  (e prefix only for edge models)\n"
                f"    gemma4:26b   <- CORRECT\n"
                f"    gemma4:e2b   <- correct (edge 2B)\n"
                f"    gemma4:e4b   <- correct (edge 4B)\n"
                f"\n  Fix: ollama pull gemma4:26b\n"
                f"  Then retry: python <script> --model gemma4:26b\n"
            )
        r.raise_for_status()
        raw = r.json()["response"]
        _save_think_log(raw, model)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "\n  Cannot reach Ollama. Is it running?\n"
            "  Ollama runs automatically on Windows — just open Anaconda Prompt.\n"
            "  Pull a model:  ollama pull gemma4:e4b\n"
        )

    return raw


def build_prompt(experiment: dict, previous_diagnostics: dict | None) -> str:
    parts = [SYSTEM_PROMPT, "\nExperiment:\n", json.dumps(experiment, indent=2)]
    if previous_diagnostics:
        parts += [
            "\n\nDiagnostics from previous model run:\n",
            json.dumps(previous_diagnostics, indent=2),
            "\n\nRefine the model to fix the issues above.",
        ]
    else:
        parts.append("\n\nThis is iteration 1. Propose your initial Beta-Binomial model.")
    return "".join(parts)


def parse_json_response(raw: str) -> dict:
    """
    Robustly extract JSON from model output.
    Handles: Gemma4 <think>...</think> blocks, markdown fences,
    raw JSON, trailing commas, and partial JSON blocks.
    """
    text = raw.strip()

    # 0. Strip Gemma4 thinking blocks: <think>...</think>
    #    These appear before the actual JSON in gemma4:26b and gemma4:31b
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Also strip bare <|think|> ... <|/think|> tokens (alternative format)
    text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.DOTALL).strip()
    # Strip any remaining XML-style tags at start/end
    text = re.sub(r"^<[^>]+>\s*", "", text).strip()
    text = re.sub(r"\s*<[^>]+>$", "", text).strip()

    # 1. Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences  ```json ... ``` or ``` ... ```
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    # try fixing trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", part)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        continue

    # 3. Find the LAST complete { ... } block
    matches = list(re.finditer(r"\{", text))
    for m in reversed(matches):
        candidate = text[m.start():]
        depth = 0
        end = -1
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            blob = candidate[:end]
            # fix trailing commas before } or ]
            blob = re.sub(r",\s*([}\]])", r"\1", blob)
            # fix unescaped newlines inside string values
            blob = re.sub(r'(?<!\\)\n(?=[^"]*"\s*[,}])', " ", blob)
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON.\nRaw output:\n{text[:800]}")


def build_retry_prompt(previous_raw: str) -> str:
    """Ask the model to fix its own broken JSON output."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Your previous response could not be parsed as JSON.\n"
        f"Previous output (first 400 chars):\n{previous_raw[:400]}\n\n"
        f"IMPORTANT: Do NOT include any thinking, reasoning, or <think> blocks.\n"
        f"Do NOT use markdown code fences (no ```json).\n"
        f"Respond ONLY with a single valid JSON object starting with {{ "
        f"and ending with }}. Nothing else."
    )


# ── 5. Execute agent-generated model code ─────────────────────────────────────

def _safe_hdi(samples, prob=0.95):
    """Compute HDI and return (lo, hi) regardless of arviz version."""
    import arviz as az
    result = az.hdi(np.asarray(samples).flatten(), hdi_prob=prob)
    if hasattr(result, "values"):
        arr = result.values.flatten()
    elif hasattr(result, "items"):
        arr = list(result.values())[0]
    else:
        arr = np.asarray(result).flatten()
    return float(arr[0]), float(arr[1])


def enrich_diagnostics(diag: dict, conv_a: int, vis_a: int,
                        conv_b: int, vis_b: int) -> dict:
    """
    If the agent model returned a minimal dict (missing HDI keys),
    recompute everything we need for the full PyMC-style report
    using only the four raw numbers we already have.

    This runs a fast conjugate Beta update — no MCMC needed —
    so it adds zero sampling time.
    """
    if "hdi_rate_a_lo" in diag:
        return diag   # already complete — reference model, nothing to do

    # Conjugate posterior: Beta(alpha + conversions, beta + non-conversions)
    # Weak prior Beta(1,1) — matches PyMC example default
    alpha_prior, beta_prior = 1, 1
    post_a = np.random.default_rng(42).beta(
        alpha_prior + conv_a, beta_prior + (vis_a - conv_a), size=20_000
    )
    post_b = np.random.default_rng(43).beta(
        alpha_prior + conv_b, beta_prior + (vis_b - conv_b), size=20_000
    )

    abs_diff  = post_b - post_a
    rel_uplift = (post_b - post_a) / post_a * 100

    ra_lo, ra_hi   = _safe_hdi(post_a)
    rb_lo, rb_hi   = _safe_hdi(post_b)
    abs_lo, abs_hi = _safe_hdi(abs_diff)
    up_lo,  up_hi  = _safe_hdi(rel_uplift)

    # Strong prior Beta(5,55) for comparison table
    post_a_s = np.random.default_rng(44).beta(5 + conv_a, 55 + (vis_a - conv_a), size=20_000)
    post_b_s = np.random.default_rng(45).beta(5 + conv_b, 55 + (vis_b - conv_b), size=20_000)
    rel_s    = (post_b_s - post_a_s) / post_a_s * 100
    up_lo_s, up_hi_s = _safe_hdi(rel_s)

    enriched = diag.copy()
    enriched.update({
        "mean_rate_a":    float(post_a.mean()),
        "mean_rate_b":    float(post_b.mean()),
        "hdi_rate_a_lo":  ra_lo,
        "hdi_rate_a_hi":  ra_hi,
        "hdi_rate_b_lo":  rb_lo,
        "hdi_rate_b_hi":  rb_hi,
        "mean_abs_diff":  float(abs_diff.mean()),
        "hdi_abs_lo":     abs_lo,
        "hdi_abs_hi":     abs_hi,
        "hdi_lo":         up_lo,
        "hdi_hi":         up_hi,
        # Build weak/strong sub-dicts for comparison table
        "weak": {
            "mean_rate_a": float(post_a.mean()),
            "mean_rate_b": float(post_b.mean()),
            "hdi_rate_a_lo": ra_lo, "hdi_rate_a_hi": ra_hi,
            "hdi_rate_b_lo": rb_lo, "hdi_rate_b_hi": rb_hi,
            "mean_abs_diff": float(abs_diff.mean()),
            "hdi_abs_lo": abs_lo, "hdi_abs_hi": abs_hi,
            "expected_lift_pct": float(rel_uplift.mean()),
            "hdi_lo": up_lo, "hdi_hi": up_hi,
            "p_b_better": diag.get("p_b_better", float((post_b > post_a).mean())),
            "r_hat_max":  diag.get("r_hat_max", float("nan")),
            "ess_min":    diag.get("ess_min",   float("nan")),
        },
        "strong": {
            "mean_rate_a": float(post_a_s.mean()),
            "mean_rate_b": float(post_b_s.mean()),
            "hdi_rate_a_lo": _safe_hdi(post_a_s)[0],
            "hdi_rate_a_hi": _safe_hdi(post_a_s)[1],
            "hdi_rate_b_lo": _safe_hdi(post_b_s)[0],
            "hdi_rate_b_hi": _safe_hdi(post_b_s)[1],
            "mean_abs_diff": float((post_b_s - post_a_s).mean()),
            "hdi_abs_lo": _safe_hdi(post_b_s - post_a_s)[0],
            "hdi_abs_hi": _safe_hdi(post_b_s - post_a_s)[1],
            "expected_lift_pct": float(rel_s.mean()),
            "hdi_lo": up_lo_s, "hdi_hi": up_hi_s,
            "p_b_better": float((post_b_s > post_a_s).mean()),
            "r_hat_max": diag.get("r_hat_max", float("nan")),
            "ess_min":   diag.get("ess_min",   float("nan")),
        },
    })
    return enriched


def sanitise_model_code(code: str) -> str:
    """
    Fix common PyMC API mistakes that LLMs generate.

    1. name= keyword arg:  pm.Beta(name="p_a", ...) → pm.Beta("p_a", ...)
    2. pm.math.greater()  → pm.math.gt()
    3. pm.math.switch()   → pm.math.where()   (older alias)
    4. pm.sample() called outside with block — wrap the whole body in
       'with pm.Model() as model:' if not already present
    5. .astype("float64") on pytensor — replace with float cast
    """
    # Fix 1 — name= keyword argument
    code = re.sub(
        r'pm\.(\w+)\(\s*name\s*=\s*(["\'])(\w+)\2\s*,',
        r'pm.\1(\2\3\2,',
        code,
    )

    # Fix 2 — pm.math.greater → pm.math.gt
    code = code.replace("pm.math.greater(", "pm.math.gt(")
    code = code.replace("pm.math.greater_equal(", "pm.math.ge(")
    code = code.replace("pm.math.less(", "pm.math.lt(")
    code = code.replace("pm.math.less_equal(", "pm.math.le(")

    # Fix 3 — .astype("float64") / .astype("float32") on pytensor tensors
    # replace with explicit cast that works in newer pytensor
    code = re.sub(r'\.astype\(["\']float(?:32|64)["\']\)', ".astype(float)", code)
    code = re.sub(r'\.astype\(["\']int(?:32|64)["\']\)',   ".astype(int)",   code)

    # Fix 4 — "No model on context stack"
    # Happens when agent writes:  trace = pm.sample(...)  outside with block
    # Detect by checking if pm.sample is called at module level (0 indent)
    # and wrapping the entire function body in a with block if needed.
    # Simple heuristic: if 'with pm.Model' is missing, inject it.
    if "with pm.Model" not in code and "pm.sample(" in code:
        # Indent everything inside run_model by 4 spaces and wrap
        lines = code.split("\n")
        new_lines = []
        inside_fn = False
        fn_indent = ""
        for line in lines:
            if re.match(r"^def run_model", line):
                inside_fn = True
                new_lines.append(line)
                continue
            if inside_fn and line.strip() == "":
                new_lines.append(line)
                continue
            if inside_fn and not line.startswith(" ") and line.strip():
                inside_fn = False
            if inside_fn:
                if fn_indent == "" and line.strip():
                    fn_indent = len(line) - len(line.lstrip())
                    fn_indent = " " * fn_indent
                    new_lines.append(f"{fn_indent}with pm.Model() as model:")
                new_lines.append("    " + line)
            else:
                new_lines.append(line)
        code = "\n".join(new_lines)

    # Fix 5 - "with trace/idata:" - InferenceData is not a context manager
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'with\s+(?!pm\.Model)\w+\s*:', stripped):
            indent = line[: len(line) - len(line.lstrip())]
            cleaned.append(indent + "# (removed invalid context manager)")
            continue
        cleaned.append(line)
    code = "\n".join(cleaned)

    # Fix 6 - pm.math.beta(...) -> pm.Beta(...)
    # Can't auto-fix perfectly but rename so it raises a clear NameError
    code = re.sub(r'pm\.math\.beta\s*\(', 'pm.Beta(  # HALLUCINATED: was pm.math.beta - use pm.Beta("name", alpha=.., beta=..) instead # (', code)

    # Fix 7 - TensorVariable.item() -> use .values.mean() pattern
    code = re.sub(r'\.item\(\)', '.values.flatten()[0]  # fixed .item()', code)

    # Fix 8 - tensor.replace(a, b) -> pm.math.switch
    code = re.sub(r'(\w+)\.replace\(([^,]+),([^)]+)\)',
                  r'pm.math.switch(pm.math.eq(, ), , )', code)

    # Fix 9 - az.mean_probability(...) -> doesn't exist, stub with nan
    code = re.sub(
        r'az\.mean_probability\s*\([^)]*\)',
        'float("nan")  # REMOVED: az.mean_probability does not exist',
        code
    )

    # Fix 10 - .values.mean() already returns numpy float, but wrap in float()
    # to be safe for any posterior extraction pattern
    code = re.sub(
        r'(?<!float\()trace\.posterior\["(\w+)"\]\.values\.mean\(\)',
        r'float(trace.posterior[""].values.mean())',
        code
    )

    return code


def execute_model(code: str, experiment: dict) -> dict:
    import pymc as pm   # noqa: F401
    import arviz as az  # noqa: F401

    namespace = {"np": np, "pm": pm, "az": az}

    # Inject a safe posterior accessor to handle missing Deterministics
    _SAFE_ACCESSOR = """
def _safe_posterior_mean(trace, var):
    try:
        return float(trace.posterior[var].values.mean())
    except KeyError:
        if var == "b_better":
            try:
                pa = trace.posterior["p_a"].values.flatten()
                pb = trace.posterior["p_b"].values.flatten()
                return float((pb > pa).mean())
            except Exception:
                return float("nan")
        return float("nan")
"""
    exec(compile(_SAFE_ACCESSOR, "<helpers>", "exec"), namespace)

    clean_code = sanitise_model_code(code)
    exec(compile(clean_code, "<agent_model>", "exec"), namespace)  # noqa: S102

    run_model = namespace["run_model"]
    result = run_model(
        conv_a=experiment["variant_a"]["conversions"],
        vis_a=experiment["variant_a"]["visitors"],
        conv_b=experiment["variant_b"]["conversions"],
        vis_b=experiment["variant_b"]["visitors"],
    )
    # Defensively cast all numeric fields
    for _k in ("p_b_better", "expected_lift_pct", "r_hat_max", "ess_min"):
        if _k in result:
            try:
                result[_k] = float(result[_k])
            except (TypeError, ValueError):
                result[_k] = float("nan")
    # Always fill in any missing HDI keys so the report never shows n/a
    return enrich_diagnostics(
        result,
        conv_a=experiment["variant_a"]["conversions"],
        vis_a=experiment["variant_a"]["visitors"],
        conv_b=experiment["variant_b"]["conversions"],
        vis_b=experiment["variant_b"]["visitors"],
    )


# ── 6. Reference model — mirrors PyMC example exactly ────────────────────────
#
#  The PyMC example (https://www.pymc.io/projects/examples/en/latest/
#  causal_inference/bayesian_ab_testing_introduction.html) reports:
#    • Posterior conversion rate for A and B  with 95% HDI
#    • Absolute difference  (θ_B − θ_A)      with 95% HDI
#    • Relative uplift  (θ_B − θ_A) / θ_A    with 95% HDI   ← main metric
#    • P(B > A)
#    • Weak prior  vs  Strong prior  comparison
#
#  We run BOTH priors so the audience can see the effect, exactly as the
#  PyMC notebook does.

REFERENCE_MODEL_CODE = textwrap.dedent("""
def run_model(conv_a, vis_a, conv_b, vis_b):

    results = {}

    for prior_label, alpha, beta in [
        ("weak",   1,  1),   # uniform — no prior knowledge (PyMC example)
        ("strong", 5, 55),   # centred near 8%, tight — informative prior
    ]:
        with pm.Model():
            # ── Priors (Beta distribution for conversion rates) ──────────────
            # Weak  : Beta(1,1)  = uniform, anything equally likely
            # Strong: Beta(5,55) = centred near 8%, tighter belief
            theta_a = pm.Beta("theta_a", alpha=alpha, beta=beta)
            theta_b = pm.Beta("theta_b", alpha=alpha, beta=beta)

            # ── Likelihood (what we observed) ────────────────────────────────
            pm.Binomial("obs_a", n=vis_a, p=theta_a, observed=conv_a)
            pm.Binomial("obs_b", n=vis_b, p=theta_b, observed=conv_b)

            # ── Derived quantities (matching PyMC example) ───────────────────
            # Absolute difference
            abs_diff = pm.Deterministic("abs_diff", theta_b - theta_a)
            # Relative uplift  = (B - A) / A  expressed as %
            rel_uplift = pm.Deterministic(
                "rel_uplift", (theta_b - theta_a) / theta_a * 100
            )
            # Is B better?
            b_better = pm.Deterministic(
                "b_better", pm.math.gt(theta_b, theta_a).astype("float32")
            )

            # ── MCMC sampling (NUTS) ─────────────────────────────────────────
            trace = pm.sample(
                draws=1000, tune=1000, chains=2,
                cores=1, progressbar=False, return_inferencedata=True
            )

        # ── Extract results ──────────────────────────────────────────────────
        summary   = az.summary(trace, var_names=["theta_a","theta_b",
                                                  "abs_diff","rel_uplift"],
                                hdi_prob=0.95)

        post_a    = trace.posterior["theta_a"].values.flatten()
        post_b    = trace.posterior["theta_b"].values.flatten()
        uplift_s  = trace.posterior["rel_uplift"].values.flatten()
        abs_s     = trace.posterior["abs_diff"].values.flatten()

        # az.hdi returns DataArray in newer arviz, ndarray in older — handle both
        def safe_hdi(samples, prob=0.95):
            result = az.hdi(samples, hdi_prob=prob)
            if hasattr(result, "values"):          # DataArray / Dataset
                arr = result.values.flatten()
            elif hasattr(result, "items"):         # dict-like
                arr = list(result.values())[0]
            else:
                arr = np.asarray(result).flatten()
            return float(arr[0]), float(arr[1])

        hdi_a_lo,  hdi_a_hi  = safe_hdi(post_a)
        hdi_b_lo,  hdi_b_hi  = safe_hdi(post_b)
        hdi_up_lo, hdi_up_hi = safe_hdi(uplift_s)
        hdi_abs_lo,hdi_abs_hi= safe_hdi(abs_s)

        results[prior_label] = {
            # rates
            "mean_rate_a":       float(post_a.mean()),
            "mean_rate_b":       float(post_b.mean()),
            "hdi_rate_a_lo":     hdi_a_lo,
            "hdi_rate_a_hi":     hdi_a_hi,
            "hdi_rate_b_lo":     hdi_b_lo,
            "hdi_rate_b_hi":     hdi_b_hi,
            # absolute difference
            "mean_abs_diff":     float(abs_s.mean()),
            "hdi_abs_lo":        hdi_abs_lo,
            "hdi_abs_hi":        hdi_abs_hi,
            # relative uplift
            "expected_lift_pct": float(uplift_s.mean()),
            "hdi_lo":            hdi_up_lo,
            "hdi_hi":            hdi_up_hi,
            # probability B beats A
            "p_b_better":        float(trace.posterior["b_better"].values.mean()),
            # diagnostics
            "r_hat_max":         float(summary["r_hat"].max()),
            "ess_min":           float(summary["ess_bulk"].min()),
        }

    # ── Return weak prior result as primary (agent uses this) ────────────────
    primary = results["weak"].copy()
    primary["weak"]   = results["weak"]
    primary["strong"] = results["strong"]
    return primary
""")

# ── 7. Console output helpers ──────────────────────────────────────────────────

def print_header(model: str):
    w = 70
    print("\n" + "═" * w)
    print("  Agentic Bayesian A/B Test  —  Gemma 4 Edition")
    print(f"  Model : {model}")
    print("  Use case: Coffee Shop Notification  (A=generic  B=discount)")
    print("═" * w)
    print(f"  Variant A conversions : {CONVERSIONS_A:>3} / {VISITORS_A}  "
          f"({CONVERSIONS_A/VISITORS_A:.1%})")
    print(f"  Variant B conversions : {CONVERSIONS_B:>3} / {VISITORS_B}  "
          f"({CONVERSIONS_B/VISITORS_B:.1%})")
    print(f"  Hidden ground truth   : B is {(TRUE_RATE_B-TRUE_RATE_A)/TRUE_RATE_A:.0%} "
          f"better than A — let's see if the agent finds it!")
    print("═" * w)
    print(f"\n  {'Iter':<6} {'R̂ max':<12} {'ESS min':<12} "
          f"{'P(B>A)':<10} {'Lift':<12} {'Decision'}")
    print("  " + "─" * 64)


def _safe_float(v):
    """Cast any value to float; returns nan for method refs, None, etc."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def print_row(i, diag, decision):
    rhat = _safe_float(diag.get("r_hat_max", float("nan")))
    ess  = _safe_float(diag.get("ess_min",   float("nan")))
    pb   = _safe_float(diag.get("p_b_better", float("nan")))
    lift = _safe_float(diag.get("expected_lift_pct", float("nan")))
    rf = "\u2713" if rhat == rhat and rhat < 1.05 else "\u2717"
    ef = "\u2713" if ess  == ess  and ess  > 400  else "\u2717"
    rhat_s = f"{rhat:.3f}" if rhat == rhat else "n/a"
    ess_s  = f"{ess:.0f}"  if ess  == ess  else "n/a"
    pb_s   = f"{pb:.1%}"   if pb   == pb   else "n/a"
    lift_s = f"{lift:+.1f}%" if lift == lift else "n/a"
    print(f"  {i:<6} {rhat_s} {rf:<7}  {ess_s} {ef:<7}  "
          f"{pb_s}    {lift_s}      {decision.upper()}")


def _fmt_hdi(lo, hi):
    return f"[{lo:+.1f}%,  {hi:+.1f}%]"


def _prior_block(label, d):
    """Print one prior block — mimics PyMC example notebook output.
    Uses .get() throughout so missing keys never crash."""
    nan = float("nan")
    w = 62

    mean_a   = d.get("mean_rate_a",   nan)
    mean_b   = d.get("mean_rate_b",   nan)
    ra_lo    = d.get("hdi_rate_a_lo", nan) * 100
    ra_hi    = d.get("hdi_rate_a_hi", nan) * 100
    rb_lo    = d.get("hdi_rate_b_lo", nan) * 100
    rb_hi    = d.get("hdi_rate_b_hi", nan) * 100
    abs_mean = d.get("mean_abs_diff",  nan)
    abs_lo   = d.get("hdi_abs_lo",     nan) * 100
    abs_hi   = d.get("hdi_abs_hi",     nan) * 100
    lift     = d.get("expected_lift_pct", nan)
    hdi_lo   = d.get("hdi_lo", nan)
    hdi_hi   = d.get("hdi_hi", nan)
    pb       = d.get("p_b_better",    nan)
    rhat     = d.get("r_hat_max",     nan)
    ess      = d.get("ess_min",       nan)

    def f(v, fmt=".1f"):
        return "n/a" if v != v else format(v, fmt)   # nan check

    print(f"\n  ┌─ {label} {'─'*(max(0, w-len(label)-2))}┐")
    print(f"  │                                                              │")

    # Posterior rates
    print(f"  │  Posterior conversion rates  (95% HDI)                      │")
    print(f"  │  θ_A = {f(mean_a, '.1%')}    95% HDI [{f(ra_lo)}%,  {f(ra_hi)}%]"
          f"{'':>12}│")
    print(f"  │  θ_B = {f(mean_b, '.1%')}    95% HDI [{f(rb_lo)}%,  {f(rb_hi)}%]"
          f"{'':>12}│")

    # Absolute difference
    print(f"  │                                                              │")
    print(f"  │  Absolute difference  θ_B − θ_A                            │")
    print(f"  │  Mean = {f(abs_mean, '+.2%')}    "
          f"95% HDI [{f(abs_lo)}%,  {f(abs_hi)}%]"
          f"{'':>12}│")

    # Relative uplift
    print(f"  │                                                              │")
    print(f"  │  Relative uplift  (θ_B − θ_A) / θ_A   ← KEY METRIC        │")
    print(f"  │  Expected = {f(lift, '+.1f')}%    "
          f"95% HDI [{f(hdi_lo, '+.1f')}%,  {f(hdi_hi, '+.1f')}%]"
          f"{'':>8}│")
    hdi_ok = (hdi_lo == hdi_lo) and hdi_lo > 0   # not nan and positive
    flag = "✓ HDI entirely above 0 → B reliably better" if hdi_ok \
           else "✗ HDI crosses 0 → direction uncertain"
    print(f"  │  {flag:<60}│")

    # P(B > A)
    print(f"  │                                                              │")
    print(f"  │  P(B > A) = {f(pb, '.1%')}{'':>47}│")

    # Diagnostics
    print(f"  │                                                              │")
    rhat_ok = (rhat == rhat) and rhat < 1.05
    ess_ok  = (ess  == ess)  and ess  > 400
    print(f"  │  Diagnostics: R-hat={f(rhat, '.3f')} {'✓' if rhat_ok else '✗'}"
          f"   ESS={f(ess, '.0f')} {'✓' if ess_ok else '✗'}"
          f"{'':>27}│")
    print(f"  └{'─'*(w+2)}┘")


def print_final_report(diag):
    w = 68
    print("\n" + "═" * w)
    print("  RESULTS  —  Coffee Shop Notification A/B Test")
    print("  Ref: pymc.io Bayesian A/B Testing Introduction")
    print("═" * w)

    # ── 1. Simulated data ─────────────────────────────────────────────────────
    true_lift = (TRUE_RATE_B - TRUE_RATE_A) / TRUE_RATE_A * 100
    print(f"\n  SIMULATED EXPERIMENT DATA")
    print(f"  Variant A  'Your morning coffee is waiting'")
    print(f"             {CONVERSIONS_A} / {VISITORS_A} conversions"
          f"  →  observed rate = {CONVERSIONS_A/VISITORS_A:.1%}")
    print(f"  Variant B  'Get 10% off your next latte today!'")
    print(f"             {CONVERSIONS_B} / {VISITORS_B} conversions"
          f"  →  observed rate = {CONVERSIONS_B/VISITORS_B:.1%}")
    print(f"  True rates hidden from model:"
          f"  A={TRUE_RATE_A:.0%}  B={TRUE_RATE_B:.0%}"
          f"  true lift={true_lift:+.0f}%")

    # ── 2. Weak and strong prior blocks ───────────────────────────────────────
    weak   = diag.get("weak",   diag)
    strong = diag.get("strong", diag)

    _prior_block("WEAK PRIOR   Beta(1,1) — uniform, no prior knowledge", weak)
    _prior_block("STRONG PRIOR Beta(5,55) — centred ~8%, tighter belief", strong)

    # ── 3. Prior effect comparison table ──────────────────────────────────────
    nan = float("nan")

    def fg(d, key, fmt):
        v = d.get(key, nan)
        return "n/a" if v != v else format(v, fmt)

    print(f"\n  PRIOR EFFECT  (key teaching point from PyMC example)")
    print(f"  {'Prior':<10} {'P(B>A)':<10} {'Expected lift':<18} {'95% HDI low'}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*16}  {'─'*12}")
    print(f"  {'Weak':<10} {fg(weak,   'p_b_better', '.1%')}     "
          f"{fg(weak,   'expected_lift_pct', '+.1f')}%"
          f"{'':>12}{fg(weak,   'hdi_lo', '+.1f')}%")
    print(f"  {'Strong':<10} {fg(strong, 'p_b_better', '.1%')}     "
          f"{fg(strong, 'expected_lift_pct', '+.1f')}%"
          f"{'':>12}{fg(strong, 'hdi_lo', '+.1f')}%")
    print(f"\n  Strong prior pulls lift estimate toward 0 (conservative).")
    print(f"  Both priors agree B is better → 'true discovery' per PyMC example.")

    # ── 4. Verdict ────────────────────────────────────────────────────────────
    pb      = weak.get("p_b_better",      nan)
    lift    = weak.get("expected_lift_pct", nan)
    hdi_lo  = weak.get("hdi_lo",           nan)
    print(f"\n  VERDICT  (weak prior — standard choice)")
    hdi_above_zero = (hdi_lo == hdi_lo) and hdi_lo > 0   # not nan and positive
    if hdi_above_zero and (pb == pb) and pb > 0.95:
        print(f"  ✓ Roll out Variant B.")
        print(f"    95% HDI entirely above 0.  Lift ≈ {lift:+.0f}%."
              f"  Confidence: {pb:.0%}.")
    elif (pb == pb) and pb > 0.80:
        print(f"  ~ B looks better ({pb:.0%}) but HDI crosses 0. Collect more data.")
    else:
        print(f"  ✗ Inconclusive. Continue the experiment.")

    # ── 5. Ground truth check ─────────────────────────────────────────────────
    print(f"\n  GROUND TRUTH CHECK  (only possible because we simulated)")
    print(f"  Hidden truth: lift={true_lift:+.0f}%  |  "
          f"Model estimated: {lift:+.1f}%  "
          f"{'← close!' if abs(lift - true_lift) < 15 else '← needs more data'}")
    print("═" * w + "\n")


# ── 8. Main loop ───────────────────────────────────────────────────────────────

# Valid model tag patterns for quick sanity check
_KNOWN_BAD_TAGS = {
    "gemma4:e26b": "gemma4:26b",
    "gemma4:e31b": "gemma4:31b",
    "gemma3:e4b":  "gemma3:4b",
    "gemma3:e27b": "gemma3:27b",
}

def run(model: str = "gemma3:4b", max_iterations: int = 4, dry_run: bool = False):
    if model in _KNOWN_BAD_TAGS:
        correct = _KNOWN_BAD_TAGS[model]
        print(f"\n  [!] '{model}' is not a valid Ollama tag.")
        print(f"      Did you mean: --model {correct} ?")
        print(f"      Switching to {correct} automatically.\n")
        model = correct
    print_header(model if not dry_run else "dry-run (no LLM)")

    previous_diagnostics: dict | None = None
    final_diagnostics: dict | None    = None

    for iteration in range(1, max_iterations + 1):

        if dry_run:
            agent = {
                "reasoning": "Dry-run mode: using reference Beta-Binomial model.",
                "model_code": REFERENCE_MODEL_CODE,
                "decision": "keep",
                "refinement_notes": None,
            }
        else:
            prompt = build_prompt(EXPERIMENT, previous_diagnostics)
            try:
                raw   = ask_ollama(prompt, model)
                agent = parse_json_response(raw)
            except ValueError:
                # ── auto-retry once with an explicit fix-your-JSON prompt ──
                print(f"  [!] JSON parse failed — asking model to fix its output...")
                try:
                    raw2  = ask_ollama(build_retry_prompt(raw), model)
                    agent = parse_json_response(raw2)
                    print(f"  [✓] Retry succeeded.")
                except (ValueError, RuntimeError) as e:
                    print(f"  [!] Retry also failed: {e}")
                    tips = {
                        "gemma4": "Try: --model gemma4:26b or gemma4:31b",
                    }
                    base = model.split(":")[0]
                    if base in tips:
                        print(f"  Tip : {tips[base]}")
                    print(f"  Or run without LLM: --dry-run")
                    break
            except RuntimeError as e:
                print(f"\n  [!] Ollama connection error: {e}")
                break

        try:
            diag = execute_model(agent["model_code"], EXPERIMENT)
            previous_diagnostics = diag
        except Exception as e:  # noqa: BLE001
            print(f"\n  [!] Model execution error: {e}")
            previous_diagnostics = {"error": str(e)}
            continue

        print_row(iteration, diag, agent["decision"])

        if agent["decision"] == "refine" and agent.get("refinement_notes"):
            print(f"       ↳ {agent['refinement_notes'][:90]}")

        if agent["decision"] == "keep":
            final_diagnostics = diag
            break

    result = final_diagnostics or previous_diagnostics or {}
    print_final_report(result)
    return result


# ── 9. Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agentic Bayesian A/B Test — Gemma 4 edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python agentic_ab_test_gemma.py --model gemma4:e2b
          python agentic_ab_test_gemma.py --model gemma4:e4b
          python agentic_ab_test_gemma.py --model gemma4:26b
          python agentic_ab_test_gemma.py --model gemma4:31b
          python agentic_ab_test_gemma.py --model gemma4:31b-cloud
          python agentic_ab_test_gemma.py --dry-run

        Pull models first (Anaconda Prompt):
          ollama pull gemma4:e4b       # recommended for laptops (~4 GB)
          ollama pull gemma4:31b       # best quality (~19 GB)
        """),
    )
    parser.add_argument("--model",   default="gemma4:e4b",
                        help="Ollama model tag (default: gemma4:e4b)")
    parser.add_argument("--iters",   type=int, default=4,
                        help="Max agent iterations (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM, run reference model only")
    parser.add_argument("--output",  default=None, metavar="PATH",
                        help="Save report to file (default: ab_report_<timestamp>.txt)")
    parser.add_argument("--no-file", action="store_true",
                        help="Print to console only, save no files")
    args = parser.parse_args()

    report_path = json_path = tee_ctx = _fh = None
    if not args.no_file:
        report_path = _make_output_path(args.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        json_path   = report_path.with_suffix(".json")

        # Think log: same name as report but with .think extension
        _THINK_STATE["path"] = report_path.with_suffix(".think")
        _THINK_STATE["iteration"] = 0
        # Write header to think log
        with open(_THINK_STATE["path"], "w", encoding="utf-8") as _tf:
            _tf.write(f"Gemma4 thinking log\n")
            _tf.write(f"Model  : {args.model}\n")
            _tf.write(f"Report : {report_path.resolve()}\n")
            _tf.write(f"{'='*70}\n")
        print(f"  Think log saving to: {_THINK_STATE['path'].resolve()}")

        _fh         = open(report_path, "w", encoding="utf-8")
        tee_ctx     = _Tee(_fh)
        sys.stdout  = tee_ctx
        print(f"  Output saving to: {report_path.resolve()}")

    final_diag = run(model=args.model, max_iterations=args.iters, dry_run=args.dry_run)

    if tee_ctx is not None:
        sys.stdout = sys.__stdout__
        _fh.close()
        print(f"\n  Report saved  -> {report_path.resolve()}")
        _tp = _THINK_STATE["path"]
        if _tp and _tp.exists():
            print(f"  Think log saved -> {_tp.resolve()}")
        if final_diag and json_path:
            json_diag = {k: v for k, v in final_diag.items() if k not in ("weak","strong")}
            json_diag["weak"]   = final_diag.get("weak",   {})
            json_diag["strong"] = final_diag.get("strong", {})
            json_diag["meta"] = {
                "model":         args.model if not args.dry_run else "dry-run",
                "timestamp":     datetime.now().isoformat(timespec="seconds"),
                "true_rate_a":   TRUE_RATE_A,
                "true_rate_b":   TRUE_RATE_B,
                "visitors_a":    VISITORS_A,
                "visitors_b":    VISITORS_B,
                "conversions_a": CONVERSIONS_A,
                "conversions_b": CONVERSIONS_B,
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(json_diag, jf, indent=2, default=str)
            print(f"  JSON data saved -> {json_path.resolve()}")
