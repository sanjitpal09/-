# From Prompts to Posteriors: Agentic Bayesian A/B Testing with PyMC

> **An autonomous LLM agent that reads experiment data, designs a Bayesian model in PyMC 5, runs MCMC sampling, evaluates convergence, and iteratively refines its own statistical code — all without human intervention.**

*Presented at Applied Machine Learning Conference, 2026 by Sanjit Paliwal, Principal Data Scientist, Verizon*

---

## Table of Contents

- [Overview](#overview)
- [Experiment Description](#experiment-description)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Repository Contents](#repository-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Script](#running-the-script)
- [CLI Arguments Reference](#cli-arguments-reference)
- [Output Files](#output-files)
- [Understanding the Code](#understanding-the-code)
- [Supported Models](#supported-models)
- [Frequentist vs. Bayesian](#frequentist-vs-bayesian)
- [References](#references)

---

## Overview

This project demonstrates an **agentic loop** for Bayesian A/B testing — inspired by Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) — where a locally-running Gemma 4 LLM (via Ollama) autonomously:

1. **Reads** a structured experiment specification (business context, variant data, objective)
2. **Reasons** about the appropriate Bayesian model (prior choice, model structure, derived quantities)
3. **Writes** self-contained PyMC 5 code for a Beta-Binomial model
4. **Executes** MCMC sampling using NUTS
5. **Evaluates** convergence diagnostics (R-hat, ESS)
6. **Decides** to `keep` or `refine` — and if refining, explains *why* and *how*
7. **Iterates** up to a configurable maximum, then prints a full statistical report

The entire loop runs locally. No external APIs are required beyond Ollama.

---

## Experiment Description

**Business context:** A coffee shop tests two push notification variants in their loyalty app to maximize purchase conversions.

| Variant | Message | True Conversion Rate |
|---------|---------|----------------------|
| **A** | *"Your morning coffee is waiting"* | 8.0% |
| **B** | *"Get 10% off your next latte today!"* | 13.0% |

Each variant is exposed to **500 users**. The ground truth is hidden from the agent; the model sees only observed conversions drawn from a binomial process. The agent's goal is to estimate P(B > A), the expected lift, and assess whether the model converged to a trustworthy posterior.

This is a canonical Bayesian A/B testing setup, replicating and extending the [PyMC Bayesian A/B Testing Introduction](https://www.pymc.io/projects/examples/en/latest/causal_inference/bayesian_ab_testing_introduction.html).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Human defines experiment + system prompt                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM (Gemma 4 via Ollama)                                   │
│  • Reads experiment JSON                                    │
│  • Reasons about prior, likelihood, derived quantities      │
│  • Writes run_model() in PyMC 5                             │
│  • Returns structured JSON: {reasoning, model_code,        │
│    decision, refinement_notes}                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Python executor                                            │
│  • Sanitises generated code (fixes common PyMC API errors)  │
│  • exec() in isolated namespace (np, pm, az only)          │
│  • Runs MCMC (NUTS), extracts diagnostics                   │
│  • Enriches diagnostics (HDI, weak/strong prior comparison) │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │ decision == "keep"? │
              └──────────┬──────────┘
            YES           NO (up to max_iters)
              │           │
              ▼           └──► Feed diagnostics back to LLM
        Final Report             and repeat
```

---

## Key Results

Results from a representative run with `gemma4:e4b` (or dry-run reference model):

| Prior | P(B > A) | Expected Lift | 95% HDI Low | Converged |
|-------|----------|--------------|-------------|-----------|
| Weak `Beta(1,1)` | **100.0%** | **+99.6%** | > 0% | ✓ |
| Strong `Beta(5,55)` | **99.9%** | **+89.0%** | > 0% | ✓ |

- **Observed conversion rate A:** 6.8% (true: 8.0%)
- **Observed conversion rate B:** 13.4% (true: 13.0%)
- **Observed lift:** 97.1% (true: 62.5%)
- Both weak and strong prior posteriors: MCMC chains fully converged (R-hat < 1.05, ESS > 400)
- **Verdict:** Roll out Variant B — the discount notification clearly outperforms the generic reminder

The strong prior pulls the lift estimate toward zero (conservative), but both priors agree B is better — a "true discovery" matching the PyMC example's methodology.

---

## Repository Contents

```
├── agentic_ab_test_gemma_8_3.py   # Main script — agentic A/B testing loop
├── Agentic_AB_Testing.pptx        # Presentation slides
└── README.md                      # This file
```

**Script internals at a glance:**

| Section | Purpose |
|---------|---------|
| Simulated data | Draws from `Binomial(500, 0.08)` and `Binomial(500, 0.13)` |
| `SYSTEM_PROMPT` | Detailed PyMC 5 API contract given to the LLM |
| `ask_ollama()` | HTTP client for Ollama `/api/generate` endpoint |
| `parse_json_response()` | Robust JSON extractor (handles `<think>` blocks, markdown fences, trailing commas) |
| `sanitise_model_code()` | Auto-corrects 10 common PyMC 3→5 API mistakes generated by LLMs |
| `execute_model()` | `exec()`s agent code in isolated namespace; enriches diagnostics |
| `REFERENCE_MODEL_CODE` | Ground-truth Beta-Binomial model (mirrors PyMC example exactly) |
| `enrich_diagnostics()` | Adds HDI, weak/strong prior sub-dicts via conjugate Beta update |
| `print_final_report()` | Full console report with prior comparison table and verdict |
| `run()` | Main agentic loop — up to `max_iterations` refinement cycles |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | f-string union types (`dict | None`) used throughout |
| PyMC | ≥ 5.0 | **Not** PyMC 3 or 4 — API is different |
| ArviZ | ≥ 0.16 | For `az.summary()`, `az.hdi()`, `az.r2_score()` |
| NumPy | ≥ 1.24 | |
| Requests | any | HTTP client for Ollama |
| Ollama | latest | [https://ollama.com](https://ollama.com) — runs locally |
| A Gemma 4 model | see below | Pull via `ollama pull` |

**Hardware:** A minimum of 8 GB RAM is recommended. For `gemma4:e4b` (~4 GB model), a modern laptop CPU is sufficient. For `gemma4:26b` or `gemma4:31b`, a GPU with 16–20+ GB VRAM is recommended.

---

## Installation

### 1. Install Ollama

Download and install from [https://ollama.com](https://ollama.com).

On Windows, Ollama runs automatically in the system tray. On macOS/Linux, start it with:
```bash
ollama serve
```

### 2. Pull a Gemma 4 Model

```bash
# Smallest — good for edge / CPU-only laptops (~2 GB)
ollama pull gemma4:e2b

# Recommended for most laptops — good quality/speed tradeoff (~4 GB)
ollama pull gemma4:e4b

# High quality — requires GPU (~16 GB)
ollama pull gemma4:26b

# Best local Gemma 4 — requires GPU (~19 GB)
ollama pull gemma4:31b

# Cloud-routed — no local download required
ollama pull gemma4:31b-cloud
```

> **Common mistake:** `gemma4:e26b` and `gemma4:e31b` are **invalid** tags. The `e` prefix is only for edge models (`e2b`, `e4b`). The 26B and 31B are full models: `gemma4:26b`, `gemma4:31b`.

### 3. Set Up Python Environment

**Option A — conda (recommended):**
```bash
conda create -n bayesian_ab python=3.11
conda activate bayesian_ab
conda install -c conda-forge pymc arviz numpy
pip install requests
```

**Option B — pip:**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pymc arviz numpy requests
```

> **Note:** If you encounter Intel OpenMP / LLVM OpenMP conflicts in a conda environment, the script already sets `KMP_DUPLICATE_LIB_OK=TRUE` to suppress the warning.

---

## Running the Script

### Standard run (LLM-powered)

```bash
# Default: gemma4:e4b, up to 4 agent iterations
python agentic_ab_test_gemma_8_3.py

# Specify a model
python agentic_ab_test_gemma_8_3.py --model gemma4:e4b
python agentic_ab_test_gemma_8_3.py --model gemma4:26b
python agentic_ab_test_gemma_8_3.py --model gemma4:31b
python agentic_ab_test_gemma_8_3.py --model gemma4:31b-cloud

# Increase max agent iterations
python agentic_ab_test_gemma_8_3.py --model gemma4:e4b --iters 6
```

### Dry run (no LLM — uses reference model)

Useful for validating your PyMC environment without needing Ollama:

```bash
python agentic_ab_test_gemma_8_3.py --dry-run
```

### Save output to a specific file

```bash
# Save to a custom path
python agentic_ab_test_gemma_8_3.py --model gemma4:e4b --output results/my_run.txt

# Print to console only, save nothing
python agentic_ab_test_gemma_8_3.py --model gemma4:e4b --no-file
```

### Expected console output (abbreviated)

```
══════════════════════════════════════════════════════════════════════
  Agentic Bayesian A/B Test  —  Gemma 4 Edition
  Model : gemma4:e4b
  Use case: Coffee Shop Notification  (A=generic  B=discount)
══════════════════════════════════════════════════════════════════════
  Variant A conversions :  34 / 500  (6.8%)
  Variant B conversions :  67 / 500  (13.4%)
  Hidden ground truth   : B is 62% better than A — let's see if the agent finds it!
══════════════════════════════════════════════════════════════════════

  Iter   R̂ max       ESS min      P(B>A)     Lift         Decision
  ────────────────────────────────────────────────────────────────
  1      1.001 ✓     1823 ✓       100.0%     +99.6%       KEEP

══════════════════════════════════════════════════════════════════════
  RESULTS  —  Coffee Shop Notification A/B Test
  ...
```

---

## CLI Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | `str` | `gemma4:e4b` | Ollama model tag to use |
| `--iters` | `int` | `4` | Maximum agent refinement iterations |
| `--dry-run` | flag | `False` | Skip LLM; run reference Beta-Binomial model only |
| `--output` | `str` | `ab_report_<timestamp>.txt` | Path for the text report (directory or file) |
| `--no-file` | flag | `False` | Console output only; no files written |

---

## Output Files

When run without `--no-file`, three files are written:

| File | Contents |
|------|---------|
| `ab_report_<timestamp>.txt` | Full console output (mirrored) |
| `ab_report_<timestamp>.json` | Structured diagnostics: rates, HDI, lift, P(B>A), convergence, metadata |
| `ab_report_<timestamp>.think` | Per-iteration raw LLM output including `<think>` blocks and JSON |

The `.think` file is particularly useful for inspecting the model's chain-of-thought reasoning at each iteration.

---

## Understanding the Code

### Agent system prompt and JSON contract

The LLM receives a strict system prompt specifying:
- It must respond with **only valid JSON** (no markdown, no prose)
- It must define exactly one function: `run_model(conv_a, vis_a, conv_b, vis_b) -> dict`
- The return dict must contain: `p_b_better`, `expected_lift_pct`, `r_hat_max`, `ess_min`
- It must use **PyMC 5 API patterns** (10 common PyMC 3→5 mistakes are explicitly listed and forbidden)
- It must autonomously choose prior hyperparameters, justify the model structure, and assess convergence

### Code sanitiser (`sanitise_model_code`)

LLMs frequently hallucinate deprecated PyMC 3 patterns. The sanitiser auto-corrects:

1. `name=` keyword in distribution constructors → positional argument
2. `pm.math.greater()` → `pm.math.gt()`
3. `.astype("float64")` on tensors → `.astype(float)`
4. Missing `with pm.Model()` context manager → injected automatically
5. `with trace:` / `with idata:` (invalid in PyMC 5) → removed
6. `pm.math.beta()` → flagged (it doesn't exist; use `pm.Beta()`)
7. `TensorVariable.item()` → `.values.flatten()[0]`
8. `az.mean_probability()` → stubbed with `nan`

### Reference model

The `REFERENCE_MODEL_CODE` is a direct implementation of the [PyMC Bayesian A/B Testing Introduction](https://www.pymc.io/projects/examples/en/latest/causal_inference/bayesian_ab_testing_introduction.html), used in `--dry-run` mode and as a correctness baseline:

```python
# Weak prior: Beta(1,1) — uniform, no prior knowledge
# Strong prior: Beta(5,55) — centred near 8%, tighter
theta_a = pm.Beta("theta_a", alpha=alpha, beta=beta)
theta_b = pm.Beta("theta_b", alpha=alpha, beta=beta)
pm.Binomial("obs_a", n=vis_a, p=theta_a, observed=conv_a)
pm.Binomial("obs_b", n=vis_b, p=theta_b, observed=conv_b)
abs_diff   = pm.Deterministic("abs_diff",   theta_b - theta_a)
rel_uplift = pm.Deterministic("rel_uplift", (theta_b - theta_a) / theta_a * 100)
b_better   = pm.Deterministic("b_better",   pm.math.gt(theta_b, theta_a).astype("float32"))
```

---

## Supported Models

All models are served via [Ollama](https://ollama.com):

| Tag | Size | Notes |
|-----|------|-------|
| `gemma4:e2b` | ~2 GB | Edge model — CPU-friendly, lower quality |
| `gemma4:e4b` | ~4 GB | **Recommended starting point** — good quality/speed |
| `gemma4:26b` | ~16 GB | High quality — GPU recommended |
| `gemma4:31b` | ~19 GB | Best local quality |
| `gemma4:31b-cloud` | 0 GB local | Cloud-routed; no GPU required |

The script verifies model tags at startup and corrects common mistakes (e.g. `gemma4:e26b` → `gemma4:26b`).

---

## Frequentist vs. Bayesian

This project focuses on the Bayesian approach. Here is a summary of the distinction:

| | Frequentist | Bayesian |
|--|------------|---------|
| **Data** | Random | Fixed (observed) |
| **Parameter** | Fixed (unknown constant) | Random (has a distribution) |
| **Output** | p-value + confidence interval | P(B > A) + credible interval (HDI) |
| **Interpretation** | "Reject / don't reject H₀" | "94% chance B is better" |
| **Decision-readiness** | Indirect | Direct and intuitive |

The **Highest Density Interval (HDI)** is the Bayesian analogue of a confidence interval — it contains the most probable parameter values and has a direct probability interpretation.

---

## References

- Fonnesbeck, C., & Wiecki, T. — *Probabilistic Programming and Bayesian Computing with PyMC*, PyData London 2024. [YouTube](https://www.youtube.com/watch?v=99Rmi_CjqME)
- Google DeepMind — *Gemma 4* (2026). [deepmind.google](https://deepmind.google/models/gemma/gemma-4/)
- Karpathy, A. — *AutoResearch* (2026). [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Ondrej, D. — *The only AutoResearch tutorial you'll ever need* (2026). [YouTube](https://www.youtube.com/watch?v=uBWuKh1nZ2Y)
- PyMC Development Team — *Introduction to Bayesian A/B testing*. [pymc.io](https://www.pymc.io/projects/examples/en/latest/causal_inference/bayesian_ab_testing_introduction.html)
- Ollama — *Gemma 4 model library*. [ollama.com/library/gemma4](https://ollama.com/library/gemma4)
- Wikipedia — *Beta distribution*. [en.wikipedia.org](https://en.wikipedia.org/wiki/Beta_distribution)
- PyMC developers — *PyMC: Probabilistic programming in Python*. [pymc.io](https://www.pymc.io/welcome.html)

---

## License

This project is shared for educational and research purposes. See individual library licenses for PyMC, ArviZ, and Ollama.

