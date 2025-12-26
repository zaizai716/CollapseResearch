#!/bin/bash
# Comprehensive status check for Model Collapse experiment

echo "=== EXPERIMENT STATUS CHECK ==="
echo ""
echo "1. Current location:"
pwd
echo ""
echo "2. Disk space:"
df -h / | head -2
echo ""
echo "3. Completed generations (have best.ckpt):"
ls -la nature_exact_experiment/gen_*/best.ckpt 2>/dev/null || echo "None found"
echo ""
echo "4. Generated data files:"
ls -lh nature_exact_experiment/gen_*/generated_data*.pkl 2>/dev/null || echo "None found"
echo ""
echo "5. Generation sizes:"
du -sh nature_exact_experiment/gen_* 2>/dev/null || echo "No generations found"
echo ""
echo "6. Check critical fixes in code:"
grep -q "pad_token = tokenizer.eos_token" Zakahler-curse_recurse-b48c90a/main.py && echo "✓ Padding fix: YES" || echo "✗ Padding fix: NO"
grep -q "preds = outputs\[:, -64:\]" Zakahler-curse_recurse-b48c90a/main.py && echo "✓ Generation fix: YES" || echo "✗ Generation fix: NO"
grep -q "'default': 0,  # Set to 0" Zakahler-curse_recurse-b48c90a/main.py && echo "✓ num_workers=0: YES" || echo "✗ num_workers=0: NO"
echo ""
echo "7. Python/PyTorch versions:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch not found"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA check failed"
echo ""
echo "8. GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "GPU not accessible"
echo ""
echo "=== GENERATION STATUS ==="
for i in 0 1 2 3 4; do
    if [ -f "nature_exact_experiment/gen_$i/best.ckpt" ]; then
        size=$(du -sh "nature_exact_experiment/gen_$i/best.ckpt" 2>/dev/null | cut -f1)
        echo "Gen $i: ✓ Complete (model: $size)"
    elif [ -f "nature_exact_experiment/gen_$i/generated_data_gen$i.pkl" ]; then
        size=$(du -sh "nature_exact_experiment/gen_$i/generated_data_gen$i.pkl" 2>/dev/null | cut -f1)
        echo "Gen $i: ⏳ Has data, ready to train (data: $size)"
    elif [ -d "nature_exact_experiment/gen_$i" ]; then
        echo "Gen $i: ⚠️ Directory exists but empty"
    else
        echo "Gen $i: ✗ Not started"
    fi
done
echo ""
echo "=== RECOMMENDATIONS ==="
# Check available space
available=$(df / | awk 'NR==2 {print $4}')
available_gb=$((available / 1024 / 1024))
if [ $available_gb -lt 5 ]; then
    echo "⚠️ Low disk space! Only ${available_gb}GB available. Consider:"
    echo "   rm nature_exact_experiment/gen_*/last.ckpt"
    echo "   rm nature_exact_experiment/gen_*/generated_data*.pkl (for completed gens)"
else
    echo "✓ Disk space OK: ${available_gb}GB available"
fi

# Check if experiment can continue
if [ -f "nature_exact_experiment/gen_0/best.ckpt" ] && [ -f "nature_exact_experiment/gen_1/best.ckpt" ]; then
    echo "✓ Ready to continue experiment"
    echo ""
    echo "Next command: python3 run_nature_experiment.py"
else
    echo "⚠️ Missing required checkpoints. May need to restart experiment."
fi
echo ""
echo "==========================="