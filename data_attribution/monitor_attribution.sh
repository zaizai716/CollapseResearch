#!/bin/bash
# Monitor attribution progress without disrupting the run

RESULT_FILE="/workspace/CollapseResearch/data_attribution/attribution_results/qwen/gen_0/influence_results_39312.json"
GRAD_DIR="/workspace/CollapseResearch/data_attribution/attribution_work/gradients/qwen/gen_0"
LOG_DIR="/workspace/CollapseResearch/data_attribution/logs"

echo "=== Attribution Progress Monitor ==="
echo "Time: $(date)"
echo ""

# Check if process is running
PROCS=$(pgrep -f "MP_main.py.*qwen_gen_0" 2>/dev/null | wc -l)
if [ "$PROCS" -gt 0 ]; then
    echo "Status: RUNNING ($PROCS processes)"
    ps aux | grep -E "MP_main.py.*qwen_gen_0" | grep -v grep | head -5
else
    echo "Status: NOT RUNNING"
fi
echo ""

# Check GPU usage
echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv 2>/dev/null || echo "GPU unavailable"
echo ""

# Check memory usage
echo "=== Memory Status ==="
free -h | head -2
echo ""

# Check result file progress
echo "=== Attribution Progress ==="
if [ -f "$RESULT_FILE" ]; then
    SIZE=$(du -h "$RESULT_FILE" | cut -f1)
    echo "Result file size: $SIZE"

    PROGRESS=$(python3 -c "
import re
with open('$RESULT_FILE', 'r') as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(max(0, size - 1000))
    content = f.read()
match = re.search(r'\"finished_cnt\":\s*\"([^\"]+)\"', content)
if match:
    parts = match.group(1).split('/')
    done = int(parts[0])
    total = int(parts[1])
    pct = (done / total) * 100
    print(f'Progress: {done:,} / {total:,} ({pct:.2f}%)')
else:
    print('Progress: Unknown')
" 2>/dev/null || echo "Could not read progress")
    echo "$PROGRESS"
else
    echo "Result file not found yet"
fi
echo ""

# Check gradients
GRAD_COUNT=$(ls "$GRAD_DIR"/train_grad_*.pt 2>/dev/null | wc -l)
echo "Saved gradients: $GRAD_COUNT / 39,312 ($(echo "scale=1; $GRAD_COUNT * 100 / 39312" | bc)%)"
echo ""

# Check latest log
LATEST_LOG=$(ls -t "$LOG_DIR"/qwen_gen0_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "=== Latest Log Tail ==="
    echo "Log: $LATEST_LOG"
    tail -10 "$LATEST_LOG" 2>/dev/null
fi
