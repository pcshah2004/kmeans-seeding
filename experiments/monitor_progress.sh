#!/bin/bash
# Monitor the comprehensive experiment progress

echo "Monitoring IMDB Comprehensive Experiment"
echo "=========================================="
echo ""

LOG_FILE="imdb_comprehensive_log.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

# Count completed runs
completed=$(grep -c "Initial cost:" "$LOG_FILE")
total=35  # 7 algorithms × 5 k values

echo "Progress: $completed / $total runs completed"
echo ""

# Show current algorithm
current_algo=$(tail -20 "$LOG_FILE" | grep "Running" | tail -1)
if [ -n "$current_algo" ]; then
    echo "Current: $current_algo"
fi

echo ""
echo "Recent output:"
echo "--------------"
tail -15 "$LOG_FILE"

echo ""
echo "To watch live: tail -f $LOG_FILE"
