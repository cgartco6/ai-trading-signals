#!/bin/bash

# Performance monitoring for Dell i7
echo "📊 Monitoring AI Trading System..."

# CPU and Memory monitoring
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEMORY_USAGE=$(free | grep Mem | awk '{print $3/$2 * 100.0}')

echo "CPU Usage: $CPU_USAGE%"
echo "Memory Usage: $MEMORY_USAGE%"

# Check if system needs optimization
if (( $(echo "$CPU_USAGE > 85" | bc -l) )); then
    echo "⚠️  High CPU usage detected. Consider reducing symbol batch size."
fi

if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "⚠️  High memory usage detected. Consider reducing cache size."
fi

# Model performance monitoring
python -c "
from utils.performance import PerformanceMonitor
monitor = PerformanceMonitor()
monitor.report_status()
"
