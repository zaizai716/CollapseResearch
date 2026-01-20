# Data Attribution for Model Collapse: Research-Paper-Ready Plan

## Research Objective
**Primary Goal:** Identify the specific categories of AI-generated data that cause model collapse, and characterize what makes them problematic.

**Concrete Deliverables:**
1. A ranked list of the most collapse-inducing training samples with analysis of their common features
2. Specific token patterns/types that correlate with high harmful influence (e.g., rare tokens, repetitive n-grams, specific Unicode ranges)
3. Actionable filtering criteria: "Data with characteristics X, Y, Z should be filtered to prevent collapse"
4. Quantified thresholds: "Samples with influence score > N contribute disproportionately to collapse"

**NOT the goal:** Simply proving data attribution can find problematic samples (that's the method, not the contribution)

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

## Part 4: Ablation-Ready Structure

### Future Experiments Enabled

1. **Model Ablation**
   - Same pipeline for OPT-125M, Llama, GPT-2
   - Compare collapse patterns across architectures
   - Config: Change `base_model_name` only

2. **Data Mixing Ablation**
   - Train on X% synthetic + (100-X)% real data
   - Find threshold where collapse begins
   - Requires new data preparation scripts

3. **Filtering Ablation**
   - Remove top-K harmful samples identified by attribution
   - Retrain and measure collapse reduction
   - Direct test of attribution utility

4. **Generation Setting Ablation**
   - Vary beam width, temperature, repetition penalty
   - See if generation diversity affects collapse
   - Config: Change main.py generation settings

### Directory Structure for Ablations

```
data_attribution/
├── attribution_results/
│   ├── qwen/           # Current experiment
│   │   ├── gen_0/
│   │   ├── gen_1/
│   │   ├── gen_2/
│   │   ├── gen_3/
│   │   └── gen_4/
│   ├── opt125m/        # Future: OPT ablation
│   ├── llama/          # Future: Llama ablation
│   └── filtered/       # Future: Filtering ablation
├── analysis/
│   ├── figures/        # Paper figures
│   ├── tables/         # Statistical tables
│   └── reports/        # Analysis summaries
└── configs/
    ├── qwen_gen_*.json
    ├── opt_gen_*.json  # Future
    └── llama_gen_*.json # Future
```

---

## Part 5: Research Paper Claims

### Primary Claims (What We Will Discover)

1. **"Specific token patterns in AI-generated data cause model collapse"**
   - Deliverable: List of problematic token patterns (e.g., "repetition of token X", "rare Unicode characters", "malformed n-grams")
   - Evidence: Token-level influence analysis showing these patterns have 10x+ higher harmful influence

2. **"AI-generated data with characteristic X, Y, Z causes collapse"**
   - Deliverable: Concrete data categories (e.g., "high-repetition samples", "low-diversity samples", "samples with perplexity > threshold")
   - Evidence: Category-wise influence breakdown showing disproportionate harm

3. **"Filtering rules can prevent collapse"**
   - Deliverable: Actionable filtering criteria with thresholds
   - Example: "Filter samples where same 3-gram appears > 5 times" or "Filter samples with unique token ratio < 0.3"
   - Evidence: Simulated filtering showing these rules would remove top harmful samples

### Secondary Claims (After Ablations)

4. **"Filtering based on identified characteristics reduces collapse by X%"**
   - Requires: Filtering ablation experiment
   - Evidence: Retrain with filtered data, measure perplexity reduction

5. **"These problematic data patterns generalize across model architectures"**
   - Requires: OPT/Llama ablations
   - Evidence: Same categories harmful across different models

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
