#!/bin/bash
# Robust execution script for Qwen Gen 0 attribution
# Designed to survive SSH disconnects and prevent RunPod freezes

set -e

LOG_DIR="/workspace/CollapseResearch/data_attribution/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/qwen_gen0_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/qwen_gen0.pid"

echo "=== Qwen Gen 0 Attribution - Robust Execution ===" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Check GPU status
echo "" | tee -a "$LOG_FILE"
echo "=== GPU Status ===" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv 2>&1 | tee -a "$LOG_FILE"

# Check memory status
echo "" | tee -a "$LOG_FILE"
echo "=== Memory Status ===" | tee -a "$LOG_FILE"
free -h 2>&1 | tee -a "$LOG_FILE"

# Check existing progress
echo "" | tee -a "$LOG_FILE"
echo "=== Checking Existing Progress ===" | tee -a "$LOG_FILE"
RESULT_FILE="/workspace/CollapseResearch/data_attribution/attribution_results/qwen/gen_0/influence_results_39312.json"
if [ -f "$RESULT_FILE" ]; then
    PROGRESS=$(python3 -c "
import json
import re
with open('$RESULT_FILE', 'r') as f:
    # Read last 1000 bytes to find finished_cnt
    f.seek(0, 2)
    size = f.tell()
    f.seek(max(0, size - 1000))
    content = f.read()
match = re.search(r'\"finished_cnt\":\s*\"([^\"]+)\"', content)
if match:
    print(match.group(1))
else:
    print('Unknown')
" 2>/dev/null || echo "Unknown")
    echo "Current progress: $PROGRESS" | tee -a "$LOG_FILE"
fi

GRAD_COUNT=$(ls /workspace/CollapseResearch/data_attribution/attribution_work/gradients/qwen/gen_0/train_grad_*.pt 2>/dev/null | wc -l)
echo "Saved gradients: $GRAD_COUNT / 39312" | tee -a "$LOG_FILE"

# Change to data_attribution directory
cd /workspace/CollapseResearch/data_attribution

# Set environment variables to prevent memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Run with proper output handling
echo "" | tee -a "$LOG_FILE"
echo "=== Starting Attribution ===" | tee -a "$LOG_FILE"
echo "Config: attribution_work/configs/qwen_gen_0_config.json" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the attribution (output goes to both log and stdout)
python3 -u RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_0_config.json 2>&1 | tee -a "$LOG_FILE"

# Save PID for monitoring
echo $$ > "$PID_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Completed ===" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
