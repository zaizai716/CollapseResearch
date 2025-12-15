#!/bin/bash
# Fast recursive training test - produces real graphs with actual model degradation
# Estimated time: 1-2 hours for 3-4 generations

echo "========================================="
echo "FAST RECURSIVE TRAINING DEMONSTRATION"
echo "========================================="
echo "This will:"
echo "1. Fine-tune OPT-125M on WikiText-2 (Gen 0)"
echo "2. Generate synthetic data from Gen 0"
echo "3. Train Gen 1 on synthetic data"
echo "4. Repeat for 3-4 generations"
echo "5. Evaluate and create degradation graphs"
echo ""
echo "Estimated time: 1-2 hours"
echo "========================================="

# Make sure we're in the right directory
cd /Users/justinyu/Desktop/CS/USC_Research/Nature_Model_Collapse

# Run the pipeline
python3 run_recursive_training.py

echo ""
echo "========================================="
echo "Training complete!"
echo "Check the graphs/ folder for results:"
echo "- graphs/quantitative/: Metrics showing numerical degradation"
echo "- graphs/qualitative/: Visual quality degradation"
echo "========================================="