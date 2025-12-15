# Model Collapse Experiment Results

## Paper: "The Curse of Recursion: Training on Generated Data Makes Models Forget"

### Experiment 1: Gaussian Mixture Model (GMM) Collapse

**Status**: ✅ Completed

**Key Findings**:
- Recursive training on synthetic data leads to progressive variance collapse
- Distributions lose diversity and converge to narrow ranges
- Effect is more pronounced with smaller sample sizes

**Results Files Generated**:
1. `gmm_collapse_results.json` - Numerical results showing variance reduction over generations
2. `gmm_model_collapse_results.png` - Visualization of distribution evolution
3. `gmm_sample_size_comparison.png` - Comparison across different sample sizes

**Quantitative Results**:
- Sample size 500: Significant variance reduction observed
- Sample size 1000: More stable but still shows progressive collapse
- Both demonstrate the fundamental issue of information loss in recursive training

### Key Metrics from GMM Experiment:
- **Original variance** maintained in generation 0
- **Progressive reduction** in variance with each generation
- **L2 distance** from original distribution increases over time
- **Component means** converge toward center of distribution

### Implications:
This experiment demonstrates the fundamental mathematical principle behind model collapse:
- Each generation loses information about the tails of the distribution
- Rare events disappear from the training data
- Models become increasingly narrow and less diverse

## How to Interpret the Results:

1. **Variance Reduction**: The decreasing variance shows how each generation captures less of the original distribution's diversity

2. **L2 Distance Growth**: Increasing L2 distance from the original GMM shows how far the learned distribution drifts from reality

3. **Visual Inspection**: The PNG files show:
   - Original distribution (wide, diverse)
   - Final distribution (narrow, collapsed)
   - Progressive degradation across generations

## Next Steps:
To complete the full replication:
1. Run VAE experiments (requires PyTorch, shows visual degradation in MNIST digits)
2. Run language model experiments (requires Transformers, shows perplexity increase)

## Files Created:
- `plt_model.py` - PyTorch Lightning wrapper for language model experiments
- `run_gmm_experiments.py` - GMM collapse experiment implementation
- Result files listed above

The GMM experiment successfully replicates the core finding of the paper: **training on AI-generated data causes inevitable model collapse**.