#!/bin/bash

# System Monitoring Script for AI Trading

echo "📊 AI Trading System Monitor"
echo "=============================="

# System resources
echo ""
echo "🖥️  SYSTEM RESOURCES:"
echo "-------------------"
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | cut -d'%' -f1)

echo "CPU Usage: ${CPU_USAGE}%"
echo "Memory Usage: ${MEMORY_USAGE}%"
echo "Disk Usage: ${DISK_USAGE}%"

# Check if system needs optimization
if (( $(echo "$CPU_USAGE > 85" | bc -l) )); then
    echo "⚠️  High CPU usage detected. Consider reducing symbol batch size."
fi

if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "⚠️  High memory usage detected. Consider reducing cache size."
fi

# Process monitoring
echo ""
echo "🔍 PROCESS STATUS:"
echo "-----------------"
if pgrep -f "python main.py" > /dev/null; then
    echo "✅ Trading system is running"
    
    # Get process details
    PROCESS_INFO=$(ps aux | grep "python main.py" | grep -v grep | head -1)
    PID=$(echo $PROCESS_INFO | awk '{print $2}')
    CPU=$(echo $PROCESS_INFO | awk '{print $3}')
    MEM=$(echo $PROCESS_INFO | awk '{print $4}')
    
    echo "   PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%"
else
    echo "❌ Trading system is not running"
fi

# Model performance
echo ""
echo "🤖 MODEL PERFORMANCE:"
echo "-------------------"
python -c "
import sys
sys.path.append('.')
from utils.logger import PerformanceLogger
import os

try:
    perf_logger = PerformanceLogger()
    summary = perf_logger.get_performance_summary()
    
    if summary:
        print(f'   Total Signals: {summary[\"total_signals\"]}')
        print(f'   Accuracy: {summary[\"accuracy\"]:.1%}')
        print(f'   Symbols Tracked: {len(summary[\"by_symbol\"])}')
    else:
        print('   No performance data available')
        
except Exception as e:
    print(f'   Performance data unavailable: {e}')
"

# Signal monitoring
echo ""
echo "📈 RECENT SIGNALS:"
echo "-----------------"
if [ -d "logs" ]; then
    RECENT_SIGNALS=$(grep -h "SIGNAL_PERFORMANCE" logs/*.log 2>/dev/null | tail -5 || echo "No recent signals")
    if [ "$RECENT_SIGNALS" != "No recent signals" ]; then
        echo "$RECENT_SIGNALS" | while read line; do
            # Extract relevant information
            SYMBOL=$(echo "$line" | grep -o '"symbol":"[^"]*"' | cut -d'"' -f4)
            ACTION=$(echo "$line" | grep -o '"action":"[^"]*"' | cut -d'"' -f4)
            CONFIDENCE=$(echo "$line" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
            OUTCOME=$(echo "$line" | grep -o '"outcome":[^,]*' | cut -d':' -f2)
            
            if [ ! -z "$SYMBOL" ]; then
                echo "   ${SYMBOL} - ${ACTION} (${CONFIDENCE:.0%}) - Outcome: ${OUTCOME}"
            fi
        done
    else
        echo "   No recent signals found"
    fi
else
    echo "   Log directory not found"
fi

# System recommendations
echo ""
echo "💡 RECOMMENDATIONS:"
echo "------------------"
if (( $(echo "$CPU_USAGE > 70" | bc -l) )); then
    echo "🔧 Reduce system.max_workers in configuration"
    echo "🔧 Decrease trading.symbols_per_batch"
fi

if (( $(echo "$MEMORY_USAGE > 75" | bc -l) )); then
    echo "🔧 Reduce data.max_cache_size"
    echo "🔧 Enable models.quantization"
fi

# Check for errors
echo ""
echo "🚨 ERROR CHECK:"
echo "--------------"
if [ -d "logs" ]; then
    ERROR_COUNT=$(grep -r "ERROR" logs/*.log 2>/dev/null | wc -l || echo "0")
    WARNING_COUNT=$(grep -r "WARNING" logs/*.log 2>/dev/null | wc -l || echo "0")
    
    echo "   Errors: $ERROR_COUNT"
    echo "   Warnings: $WARNING_COUNT"
    
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "   Recent errors:"
        grep -h "ERROR" logs/*.log 2>/dev/null | tail -3 | while read error; do
            echo "   - $(echo $error | cut -d':' -f4-)"
        done
    fi
else
    echo "   No logs available"
fi

echo ""
echo "✅ Monitoring complete - $(date)"
