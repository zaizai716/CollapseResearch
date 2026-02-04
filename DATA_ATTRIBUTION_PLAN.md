# Data Attribution for Model Collapse: Research-Paper-Ready Plan

## Research Objective
**Primary Goal:** Identify the specific categories of AI-generated data that cause model collapse, and characterize what makes them problematic.

**Novelty / Why this matters:** Existing model collapse papers (Shumailov et al. Nature 2023, Alemohammad et al. 2023) demonstrate that collapse *happens* when training on AI-generated data, but they don't explain *which* data causes it or *why*. We use influence functions (RapidIn/TracIn) to perform fine-grained sample-level and token-level attribution, pinpointing the exact training samples and token patterns responsible for collapse. This is the first work to apply data attribution methods to the model collapse problem.

**Concrete Deliverables:**
1. A ranked list of the most collapse-inducing training samples with analysis of their common features
2. Specific token patterns/types that correlate with high harmful influence (e.g., rare tokens, repetitive n-grams, specific Unicode ranges)
3. Actionable filtering criteria: "Data with characteristics X, Y, Z should be filtered to prevent collapse"
4. Quantified thresholds: "Samples with influence score > N contribute disproportionately to collapse"

**NOT the goal:** Simply proving data attribution can find problematic samples (that's the method, not the contribution)

---

## Research Pipeline Overview

This research follows a 4-phase pipeline. Each phase builds on the previous one and unlocks stronger claims:

```
PHASE 1: BASELINE (current)          PHASE 2: ANALYSIS              PHASE 3: VALIDATION           PHASE 4: PAPER
─────────────────────────────         ─────────────────               ─────────────────              ──────────
Run influence attribution             Extract patterns from           Ablation experiments           Write up with
on Qwen3-1.7B across 5               attribution results.            to prove findings are          full evidence.
generations of collapse.              Characterize harmful            causal & generalizable.
                                      data categories & tokens.

PRODUCES:                             PRODUCES:                       PRODUCES:                      PRODUCES:
- Per-sample influence scores         - Harmful data taxonomy         - Causal evidence              - Publication-ready
- Per-token influence scores          - Token pattern analysis        - Cross-model validation       - Figures & tables
- Top-K helpful/harmful rankings      - Filtering rule candidates     - Filtering effectiveness      - Reproducibility pkg
                                      - Statistical correlations        measurements

CLAIMS UNLOCKED:                      CLAIMS UNLOCKED:                CLAIMS UNLOCKED:
(none yet — this is raw data)         "Samples with X cause           "Filtering rule X reduces
                                       collapse" (correlational)       collapse by Y%" (causal)
                                                                      "Patterns generalize across
                                                                       architectures" (robust)
```

**We are currently in Phase 1 (baseline).** Phase 1 alone produces no publishable claims — it produces the raw attribution data that Phase 2 analyzes. But Phase 1 must be done correctly and completely, because everything downstream depends on it.

---

## Current Status

### Completed
- Model collapse experiment: 5 generations (Gen 0-4) with Qwen3-1.7B
- Perplexity explosion: 9.8 → 37,054 (3,766x increase)
- All training data: 39,312 samples per generation (consistent)
- Attribution configs: All 5 configs validated and ready
- Data files: All exist with correct formats and sample counts
- Checkpoints: All symlinks valid

### In Progress
- Gen 0 attribution: Phase 1 (s_test) complete, Phase 2 (influence) at 26%, Phase 3 (word_influence) not started
- Gen 4 synthetic data generation

---

## Part 1: Run Attribution Experiments

### Execution Order
Run sequentially (GPU memory constraints):

```bash
cd /workspace/CollapseResearch/data_attribution

# Gen 0: Which WikiText samples influenced Gen 1 output?
python RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_0_config.json

# Gen 1: Which Gen 1 synthetic samples influenced Gen 2 output?
python RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_1_config.json

# Gen 2: Which Gen 2 synthetic samples influenced Gen 3 output?
python RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_2_config.json

# Gen 3: Which Gen 3 synthetic samples influenced Gen 4 output?
python RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_3_config.json

# Gen 4: (After synthetic data ready) Which Gen 4 samples influenced its own output?
python RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_4_config.json
```

### Output Files
Each run produces:
- `attribution_results/qwen/gen_X/influence_results_39312.json`
  - `helpful`: Top 1000 most helpful training sample IDs
  - `helpful_infl`: Corresponding influence scores
  - `harmful`: Top 1000 most harmful training sample IDs
  - `harmful_infl`: Corresponding influence scores
  - `word_influence`: Per-token influence breakdown (key for analysis)

---

## Part 2: Analysis Pipeline for Research Claims

### Analysis Script: `analyze_collapse_attribution.py`

Create this script to identify WHAT causes collapse:

```python
# Key analyses to deliver concrete findings:

1. TOP HARMFUL SAMPLES DEEP DIVE
   - Extract top-100 most harmful samples from each generation
   - Print their full text content for manual inspection
   - Compute: avg length, token freq distribution, n-gram patterns
   - Answer: "What do the worst samples look like?"

2. TOKEN-LEVEL BREAKDOWN
   - For each harmful sample, which specific tokens have highest negative influence?
   - Aggregate: Which token types appear most in harmful positions?
   - Categories to check:
     * Rare tokens (freq < threshold in training)
     * Repetitive tokens (same token appearing N+ times)
     * Special characters / Unicode
     * Punctuation patterns
     * Named entities vs common words
   - Answer: "Which specific tokens cause problems?"

3. DATA CATEGORY CLASSIFICATION
   - Classify each training sample into categories:
     * High repetition (same phrase repeated)
     * Low diversity (few unique tokens)
     * Incoherent (high perplexity on reference model)
     * Contains errors (Unicode errors, encoding issues)
     * Normal text
   - Compute influence by category
   - Answer: "Which DATA CATEGORIES cause collapse?"

4. CROSS-GENERATION PATTERN TRACKING
   - Do the same problematic patterns appear across generations?
   - Does removing category X from Gen 1 prevent collapse in Gen 2+?
   - Answer: "Is there a consistent bad data signature?"

5. ACTIONABLE FILTERING RULES
   - Derive filtering criteria from findings:
     * "Filter samples with repetition ratio > X"
     * "Filter samples with rare token ratio > Y"
     * "Filter samples with perplexity > Z"
   - Validate: Would these rules have caught the bad data?
   - Answer: "How can we filter bad AI data in practice?"
```

### Key Metrics to Report

| Metric | Description | Research Significance |
|--------|-------------|----------------------|
| Top-K Influence Ratio | Sum of top-K influences / total | Shows concentration |
| Harmful/Helpful Ratio | Count of harmful vs helpful samples | Balance indicator |
| Token Influence Variance | Variance in per-token scores | Identifies critical tokens |
| Cross-Gen Correlation | Spearman ρ between gen influence ranks | Tracks persistence |
| Perplexity-Influence Corr | Correlation with collapse metric | Validates causality |

---

## Part 3: Visualization for Paper

### Required Figures

1. **Figure: Influence Distribution by Generation**
   - Violin plots showing influence score distributions
   - Shows how influence patterns change as collapse progresses

2. **Figure: Top Influential Tokens Heatmap**
   - Heatmap of token types vs generations
   - Color = average influence score
   - Reveals which token categories drive collapse

3. **Figure: Sample Influence Trajectories**
   - Line plot tracking specific samples across generations
   - Shows if certain samples consistently harmful

4. **Figure: Token Frequency vs Influence**
   - Scatter plot with regression line
   - Tests hypothesis: rare tokens → more harmful

5. **Figure: Collapse Metric vs Attribution Summary**
   - Dual-axis plot: perplexity + influence concentration
   - Shows correlation between collapse and attribution patterns

---

## Part 4: Validation Ablations (Phase 3 — After Baseline + Analysis)

Phase 2 analysis will produce *correlational* findings ("samples with property X tend to be harmful"). Phase 3 ablations turn those into *causal* and *generalizable* claims, which is what separates a strong paper from a weak one.

### Ablation 1: Filtering Ablation (MOST IMPORTANT — proves causality)
- **What**: Take the filtering rules discovered in Phase 2, apply them to the training data, retrain the model, and measure if collapse is reduced
- **Why this matters**: Without this, reviewers will say "you found correlations, not causes." This ablation proves the attribution findings are *actionable*.
- **Design**:
  - Control: Retrain with full synthetic data (same as baseline) → expect same collapse
  - Treatment: Retrain with filtered synthetic data (remove top-K harmful samples) → expect reduced collapse
  - Metric: Compare perplexity curves across generations
- **Claim unlocked**: "Filtering based on influence attribution reduces collapse by X%"

### Ablation 2: Cross-Model Generalization (proves robustness)
- **What**: Run the same attribution pipeline on a different model architecture (e.g., OPT-125M, Llama, GPT-2)
- **Why this matters**: Shows that the harmful data patterns aren't model-specific artifacts. If the same token patterns and data categories show up across architectures, the findings are general.
- **Design**: Same pipeline, swap `base_model_name` in config
- **Claim unlocked**: "These problematic data patterns generalize across model architectures"

### Ablation 3: Data Mixing Threshold (characterizes severity)
- **What**: Train on X% synthetic + (100-X)% real data for varying X, run attribution on each
- **Why this matters**: Finds the tipping point where synthetic data starts causing collapse, and whether the same samples are identified as harmful at different mixing ratios
- **Claim unlocked**: "Collapse begins when synthetic data exceeds X% of training data, driven by the same harmful sample categories"

### Ablation 4: Generation Setting Variation (identifies root cause)
- **What**: Vary generation parameters (temperature, top-p, repetition penalty) and see if the harmful data patterns change
- **Why this matters**: If certain generation settings produce fewer harmful samples, this gives a practical mitigation strategy upstream
- **Claim unlocked**: "Generation settings that increase diversity reduce the proportion of harmful training samples by X%"

### Directory Structure for Ablations

```
data_attribution/
├── attribution_results/
│   ├── qwen/           # Phase 1 baseline (CURRENT)
│   │   ├── gen_0/
│   │   ├── gen_1/ ... gen_4/
│   ├── qwen_filtered/  # Ablation 1: Filtering
│   ├── opt125m/        # Ablation 2: Cross-model
│   ├── llama/          # Ablation 2: Cross-model
│   └── qwen_mixed/     # Ablation 3: Data mixing
├── analysis/
│   ├── figures/        # Paper figures
│   ├── tables/         # Statistical tables
│   └── reports/        # Analysis summaries
└── configs/
    ├── qwen_gen_*.json
    ├── opt_gen_*.json       # Ablation 2
    └── qwen_filtered_*.json # Ablation 1
```

---

## Part 5: Research Paper Claims — Evidence Mapping

### How Baseline → Analysis → Ablations Build the Full Story

| Claim | Strength | Requires | Phase |
|-------|----------|----------|-------|
| "Influence functions can identify training samples that drive collapse" | Descriptive (weak alone) | Baseline attribution + sanity checks | 1+2 |
| "Harmful samples share identifiable patterns (repetition, low diversity, etc.)" | Correlational | Baseline + pattern analysis | 2 |
| "Influence concentration increases across generations as collapse worsens" | Correlational | Baseline across all 5 gens | 2 |
| "Token-level attribution reveals specific token types that drive collapse" | Correlational | Baseline + word_influence analysis | 2 |
| **"Filtering harmful samples reduces collapse by X%"** | **Causal (strong)** | **Filtering ablation** | **3** |
| "Harmful data patterns generalize across model architectures" | Robustness | Cross-model ablation | 3 |
| "Collapse threshold depends on synthetic data proportion" | Characterization | Data mixing ablation | 3 |

### Primary Claims (from Baseline + Analysis — Phases 1-2)

1. **"Specific token patterns in AI-generated data cause model collapse"**
   - Evidence: Token-level influence analysis showing these patterns have 10x+ higher harmful influence
   - Strength: Correlational — strong if consistent across all 5 generations

2. **"AI-generated data with characteristic X, Y, Z causes collapse"**
   - Evidence: Category-wise influence breakdown showing disproportionate harm
   - Strength: Correlational — becomes causal with filtering ablation

3. **"Influence concentration increases as collapse progresses"**
   - Evidence: Compare influence distributions across Gen 0→4, show increasing concentration
   - Strength: Descriptive but compelling trend

### Secondary Claims (from Ablations — Phase 3, elevates paper significantly)

4. **"Filtering based on identified characteristics reduces collapse by X%"**
   - Evidence: Retrain with filtered data, measure perplexity reduction
   - Strength: **Causal** — this is the strongest possible claim and the key result reviewers will look for

5. **"These problematic data patterns generalize across model architectures"**
   - Evidence: Same categories harmful across different models
   - Strength: Robustness — addresses the "is this just a Qwen thing?" question

---

## Part 6: Execution Checklist

### Phase A: Run Attribution
- [ ] Run Gen 0 attribution
- [ ] Run Gen 1 attribution
- [ ] Run Gen 2 attribution
- [ ] Run Gen 3 attribution
- [ ] Wait for Gen 4 synthetic data
- [ ] Convert Gen 4 pkl to jsonl
- [ ] Run Gen 4 attribution

### Phase B: Analysis Pipeline
- [ ] Create analyze_collapse_attribution.py
- [ ] Implement influence concentration analysis
- [ ] Implement token-level pattern analysis
- [ ] Implement statistical validation
- [ ] Generate all figures

### Phase C: Upload & Document
- [ ] Upload attribution results to HuggingFace
- [ ] Document experimental methodology
- [ ] Create reproducibility package

---

## Config Validation Summary

All configs verified correct:

| Config | Train Data | Test Data | Status |
|--------|------------|-----------|--------|
| gen_0 | WikiText-2 (39,312) | Gen 1 synthetic | ✓ Ready |
| gen_1 | Gen 1 synthetic (39,312) | Gen 2 synthetic | ✓ Ready |
| gen_2 | Gen 2 synthetic (39,312) | Gen 3 synthetic | ✓ Ready |
| gen_3 | Gen 3 synthetic (39,312) | Gen 4 synthetic | ✓ Ready |
| gen_4 | Gen 4 synthetic (39,312) | Gen 4 output | ⏳ Waiting |

### RapidIn Settings (All Configs)
- `cal_words_infl`: true (token-level attribution)
- `RapidGrad_K`: 65536 (gradient compression)
- `shuffle_lambda`: 20 (projection quality)
- `top_k`: 1000 (samples to report)
- `seed`: 42 (reproducibility)

---

## Verification Steps

1. **Attribution Output Check**
   ```bash
   # After each run, verify output exists and has expected structure
   python -c "import json; d=json.load(open('attribution_results/qwen/gen_0/influence_results_39312.json')); print(f'Test samples: {len(d)-1}, Has word_influence: {\"word_influence\" in d[\"0\"]}')"
   ```

2. **Influence Score Sanity Check**
   - Helpful scores should be positive
   - Harmful scores should be negative
   - Scores should vary (not all identical)

3. **Cross-Generation Consistency**
   - Same sample IDs should appear across generations
   - Influence patterns should show progression

---

## Known Issues & TODOs

### OPT-125M Attribution Issues (Fix Later)
**Problem:** OPT attribution experiments have critical issues that need fixing before use.

1. **Wrong Test Data**: OPT configs use `collapse_test.jsonl` containing manually-crafted fake prompts, NOT actual model collapse output. Need to regenerate with actual Gen N synthetic output.

2. **Massive NaN Gradient Issues**: Logs show 188,669+ NaN warnings:
   ```
   Warning: Got NaN influence for sample X, setting to 0
   ```
   Root cause: `model.half()` in `engine.py:94` causes FP16 precision issues.

   **Fix options:**
   - Remove `.half()` call (use FP32)
   - Add gradient clipping
   - Use mixed precision training properly

3. **Training Data Files Missing**: The OPT training data paths no longer exist.

**Status:** Deprioritized. Focus on Qwen experiments first, fix OPT later for model architecture ablation.

### Qwen Attribution Scale
**Note:** Full Qwen gen_0 attribution requires 1.5 billion computations (39,312 × 39,312). Previous run only completed 10%. Use robust execution setup (nohup, tmux, monitoring) to prevent RunPod freezes.

---

## CURRENT SESSION STATUS (Feb 4, 2026)

### Current Progress — Gen 0 Attribution
- **Phase 1 (s_test vectors)**: COMPLETE. Cached at `s_test_cache/qwen_gen_0_s_test.pt` (4.9GB)
- **Phase 2 (influence computation)**: **26% complete** (401,812,658 / 1,545,433,344). Saved in `influence_results_39312_2026-02-03-01-40-25.json`
- **Phase 3 (word_influence)**: 0% — requires Phase 2 to finish first
- **Gradients**: Fully cached for gen_0 (21,673 files, 4.4GB in `gradients/qwen/gen_0/`)

### What's Currently Going Wrong

1. **Experiment keeps getting killed mid-run**: The influence computation (Phase 2) is a massive 1.5 billion dot-product loop (39,312 train × 39,312 test). It takes days to complete, and the process keeps dying — either from RunPod session timeouts, pod freezes, or OOM issues on multi-GPU setups.

2. **Latest resume attempt stuck during initialization**: The most recent run (`gen_0_resume_20260203_192529.log`) started on 3 GPUs, began tokenizing data, but **never progressed past tokenization**. All 3 GPU workers loaded tokenizers and started tokenizing but the process hung there — likely a memory issue trying to load 3 copies of Qwen3-1.7B simultaneously, or a multiprocessing deadlock at the `start_barrier.wait()`.

3. **Resume doesn't truly resume Phase 2 computation**: The "resume" feature loads the saved JSON and advances the progress counter, but the in-memory `infl_list` matrix (39,312 × 39,312) is re-initialized to all zeros. The workers also reset `finished_idx` to all False, so every training sample gets reprocessed. The only real time savings are:
   - s_test vectors loaded from cache (skips Phase 1 entirely)
   - Training gradients loaded from disk cache (skips gradient computation)
   - But all influence dot products are recomputed from scratch

4. **Bad 72% result file**: There is a second file `influence_results_39312.json` that claims 72% progress, but this file is **corrupt/incomplete** — it was produced by a run that didn't have all necessary parts. The valid progress checkpoint is the 26% file.

5. **Config resume_path points to 26% file**: The config (`qwen_gen_0_config.json`) has `resume_path` set to the 26% file, which is correct. But since resume doesn't actually restore the influence matrix, this mostly just affects the progress display counter.

### What Needs to Be Fixed

1. **Fix the initialization hang on multi-GPU**: The 3-GPU run hangs during tokenization/model loading. Options:
   - Try with fewer GPUs (1-2) to reduce memory pressure
   - Stagger model loading across GPUs instead of all at once
   - Add timeouts to the barrier synchronization

2. **Make Phase 2 actually resumable**: Currently the influence matrix is lost on restart. Need to either:
   - Periodically checkpoint the full `infl_list` matrix to disk (large but enables true resume)
   - Or save per-training-sample influence vectors so they can be reloaded
   - Or accept recomputation since cached gradients make the dot products fast

3. **Delete the bad 72% result file**: `influence_results_39312.json` should be removed or renamed to avoid confusion. Only the 26% file is valid.

4. **Ensure stable long-running execution**: Use `nohup`, `tmux`/`screen`, and monitoring to prevent session drops from killing the process. Consider a watchdog script that auto-restarts on crash.

### What's Already Been Fixed (Previous Sessions)
1. **s_test caching**: Modified `engine.py` to save/load s_test vectors (config: `s_test_path`, `save_s_test`, `load_s_test`)
2. **Gradient caching**: Training gradients saved to disk and reloaded on restart
3. **Helper scripts**: `setup_attribution.sh`, `run_qwen_gen0_robust.sh`, `monitor_attribution.sh`

### Backed Up to HuggingFace (Feb 4, 2026)
All partial progress uploaded to `zaizaiiiii/model-collapse-experiment` under `attribution/gen_0_partial_results/`:
- `influence_results_39312_partial_26pct.json` (950MB) — valid 26% progress
- `influence_results_39312_earlier_partial.json` (916MB) — earlier run backup
- `qwen_gen_0_s_test_cache.pt` (4.9GB) — complete s_test vectors

### Next Steps (FOR NEXT SESSION)
1. **Delete or rename the bad 72% file** to avoid confusion
2. **Fix the multi-GPU initialization hang** — try 1 GPU first to verify the run works, then scale up
3. **Update resume_path** in config to point to the correct 26% file (already set)
4. **Run the experiment in a stable session** (tmux + nohup + monitoring)
5. After Phase 2 completes, Phase 3 (word_influence) runs automatically
6. Then proceed to Gen 1-4 attribution

### Time Estimates
| Phase | 1 GPU | 3 GPUs |
|-------|-------|--------|
| Phase 1 (s_test) | DONE | DONE |
| Phase 2 (influence) — remaining 74% | ~70 hrs | ~23 hrs |
| Phase 3 (word_influence) | ~10 hrs | ~3 hrs |
| **Remaining for Gen 0** | **~80 hrs (3.3 days)** | **~26 hrs (1.1 days)** |
| **All 5 generations total** | **~20 days** | **~6 days** |

### Files That Need to Persist
- `attribution_work/gradients/qwen/gen_*/` — cached training gradients (reusable)
- `attribution_work/s_test_cache/` — cached s_test vectors (Phase 1 complete)
- `attribution_results/qwen/gen_*/` — partial and final results
