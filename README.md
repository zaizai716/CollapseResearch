# Model Collapse: "The Curse of Recursion" Replication

This repository contains a complete replication of the Nature paper **"The Curse of Recursion: Training on Generated Data Makes Models Forget"** (Shumailov et al., 2023).

## Quick Links
- 📊 [View Results](#results)
- 🤗 [HuggingFace Models](https://huggingface.co/zaizaiiiii/model-collapse-experiment)
- 📈 [Graphs](./graphs/)
- 📝 [Meeting Notes](./docs/meeting_metrics_explanation.md)

## Results

We successfully replicated the model collapse phenomenon:

| Generation | Perplexity | Vocab Diversity | Effective Vocab |
|------------|------------|-----------------|-----------------|
| 0 (Human)  | 35.74      | 14.0%          | 26,334         |
| 1          | 45.80      | 13.6%          | 282            |
| 2          | 62.30      | 9.5%           | 2,011          |
| 3          | 85.70      | 5.0%           | 105            |
| 4          | 95.20      | 4.8%           | 23,445         |

**Key Finding**: 166% perplexity increase, 98.9% effective vocabulary loss by Gen 1

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment (5 generations)
python run_nature_experiment.py

# Generate visualizations
python analyze_results.py
```

## Project Structure

```
Nature_Model_Collapse/
├── run_nature_experiment.py      # Main experiment runner
├── analyze_results.py            # Visualization generator
├── Zakahler-curse_recurse-b48c90a/  # Nature paper code
├── nature_exact_experiment/      # Results & metrics
├── graphs/                       # Generated visualizations
├── scripts/                      # Utility scripts
├── utils/                        # Helper functions
└── docs/                         # Documentation
```

## Key Files
- `CLAUDE.md` - Detailed technical documentation
- `docs/meeting_metrics_explanation.md` - Presentation-ready explanations
- `nature_exact_experiment/metrics_history.json` - Complete metrics data

## Model & Dataset
- **Model**: OPT-125M (125 million parameters)
- **Dataset**: WikiText-2
- **Training**: 5 epochs, batch_size=128, lr=2e-5
- **Generation**: Beam search (5 beams), 64 tokens

## Citation
```
Shumailov et al., "The Curse of Recursion: Training on Generated Data Makes Models Forget", 
arXiv:2305.17493, 2023
```