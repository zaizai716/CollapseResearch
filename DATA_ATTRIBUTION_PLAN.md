# Data Attribution for Model Collapse: Research-Paper-Ready Plan

## Research Objective
**Claim to support:** "Certain categories of AI-generated training data cause model collapse, and data attribution can identify these problematic samples for filtering."

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

Create this script to extract research insights:

```python
# Key analyses to implement:

1. INFLUENCE CONCENTRATION ANALYSIS
   - Are certain training samples disproportionately influential?
   - Does influence become more concentrated as collapse progresses?
   - Gini coefficient of influence scores per generation

2. TOKEN-LEVEL PATTERN ANALYSIS
   - Which tokens have highest average influence on collapse?
   - Do rare/OOV tokens correlate with harmful influence?
   - Token frequency vs influence correlation

3. CROSS-GENERATION INFLUENCE TRACKING
   - Do the same "harmful" patterns persist across generations?
   - Influence score trajectories for specific sample types
   - Feedback amplification detection

4. SAMPLE CATEGORIZATION
   - Cluster training samples by influence patterns
   - Identify "collapse-inducing" vs "stabilizing" sample categories
   - Extract common features of harmful samples

5. STATISTICAL VALIDATION
   - Bootstrap confidence intervals on influence rankings
   - Permutation tests for influence significance
   - Effect size calculations (Cohen's d)
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

### Primary Claims (Supported by Current Experiments)

1. **"Data attribution identifies training samples that cause model collapse"**
   - Evidence: Influence scores correlate with perplexity increase
   - Validation: Top-K harmful samples analysis

2. **"Collapse-inducing samples share identifiable characteristics"**
   - Evidence: Token-level analysis reveals patterns
   - Validation: Clustering of harmful samples

3. **"Influence concentration increases with collapse severity"**
   - Evidence: Gini coefficient of influence across generations
   - Validation: Statistical trend test

### Secondary Claims (After Ablations)

4. **"Filtering high-influence samples reduces collapse"**
   - Requires: Filtering ablation experiment
   - Evidence: Perplexity comparison pre/post filtering

5. **"Collapse patterns generalize across model architectures"**
   - Requires: OPT/Llama ablations
   - Evidence: Cross-model influence correlation

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
