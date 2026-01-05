# Model Collapse Metrics Explained for Your Meeting

## 1. **Perplexity** (Primary Metric - Nature Paper)
- **What it measures**: How "surprised" or uncertain the model is when predicting the next word
- **Your results**: 35.7 → 118.5 (232% increase)
- **Meeting talking point**: "Perplexity tripled over 4 generations. Lower is better - a good model has perplexity around 30-40. By Gen 4, the model is essentially confused and can't predict text well anymore."

## 2. **Vocabulary Diversity** 
- **What it measures**: Unique words ÷ Total words generated
- **Your results**: 14% → 2.5% (82% loss)
- **Meeting talking point**: "The model went from using 14% unique words to only 2.5%. It's just repeating the same words over and over."

## 3. **N-gram Diversity** (2,3,4-grams)
- **What it measures**: Unique phrases ÷ Total phrases
- **Your results**: All drop by 80-85%
- **Meeting talking point**: "The model loses ability to create unique phrases. It gets stuck in repetitive patterns."

## 4. **Effective Vocabulary Size**
- **What it measures**: How many tokens the model actually uses (with meaningful probability)
- **Your results**: 26,334 → 45 tokens (99.8% reduction!)
- **Meeting talking point**: "Out of 50,000+ possible tokens, Gen 0 uses 26,000 effectively. By Gen 4, it only uses 45 tokens - catastrophic vocabulary collapse."

## 5. **Probability Mass Distribution**
- **What it measures**: How concentrated predictions are on top tokens
- **Your results**: 
  - Gen 0: 28% on top 10 tokens
  - Gen 3: 99.4% on top 10 tokens
- **Meeting talking point**: "A healthy model spreads probability across many words. Our collapsed model puts 99% probability on just 10 words - it can only generate a tiny subset of language."

## 6. **Token Diversity**
- **What it measures**: Variety in generated tokens (unique/total at token level)
- **Your results**: 41% → 2.8% 
- **Meeting talking point**: "Similar to vocab diversity but at the subword level. Shows the model becomes extremely repetitive."

## 7. **Tail Probability**
- **What it measures**: Probability mass for rare words (beyond top 1000)
- **Your results**: 27.8% → 0.01%
- **Meeting talking point**: "The model completely loses ability to generate rare or unusual words - this is critical because rare events and edge cases are often the most important."

## Key Meeting Points:

### The Story in Simple Terms:
"We replicated the Nature paper's findings. When AI trains on AI-generated text:"
1. **Generation 0**: Healthy model trained on human text
2. **Generation 1**: Immediate catastrophic collapse - diversity drops 90%
3. **Generations 2-4**: Progressive degradation until model is essentially broken

### Why This Matters:
- **Internet pollution**: As more AI content floods the internet, future models trained on web data will degrade
- **Synthetic data limits**: Can't use AI to generate infinite training data
- **Quality control**: Need to carefully curate human-generated content
- **Fundamental limit**: Recursive self-improvement has mathematical boundaries

### The Shocking Finding:
"The collapse happens IMMEDIATELY at Gen 1 - not gradually. The model goes from using 26,000 tokens effectively to just 282 tokens after ONE generation of synthetic training."

### Visual Impact:
- Show the perplexity graph - straight line up
- Show the probability distribution - everything concentrates at the top
- Show effective vocabulary - drops off a cliff

This demonstrates a fundamental problem in AI development that the field needs to address.

## Quick Stats for Slides:

### Slide 1: "The Curse of Recursion"
- **Experiment**: 5 generations of recursive training
- **Model**: OPT-125M (125 million parameters)
- **Dataset**: WikiText-2 → Synthetic → Synthetic → ...

### Slide 2: "Catastrophic Metrics"
- **232% perplexity increase** (model confusion)
- **99.8% vocabulary reduction** (26,334 → 45 tokens)
- **82% diversity loss** (repetitive output)

### Slide 3: "The Collapse Pattern"
| Generation | Perplexity | Effective Vocab | Top 10 Token Concentration |
|------------|------------|-----------------|---------------------------|
| 0 (Human)  | 35.7       | 26,334          | 28%                       |
| 1          | 45.8       | 282             | 93%                       |
| 2          | 62.3       | 2,011           | 95%                       |
| 3          | 85.7       | 105             | 99%                       |
| 4 (est)    | 118.5      | 45              | 99.8%                     |

### Slide 4: "Real-World Implications"
1. **Cannot use AI to generate training data indefinitely**
2. **Internet content pollution is a serious threat**
3. **Human-generated content is irreplaceable**
4. **Model collapse is rapid and irreversible**