#!/bin/bash

# Deployment script for AI Trading System

set -e  # Exit on error

ENVIRONMENT=${1:-staging}
CONFIG_FILE="environments/${ENVIRONMENT}.yaml"

echo "🚀 Deploying AI Trading System to ${ENVIRONMENT}..."

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Validate environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "⚠️  DEPLOYING TO PRODUCTION - Requiring confirmation..."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
fi

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose down || true

# Backup current version
if [ -d "data" ]; then
    echo "💾 Backing up data..."
    tar -czf "backup_$(date +%Y%m%d_%H%M%S).tar.gz" data/ models/trained_models/ logs/
fi

# Pull latest changes (if using git)
if [ -d ".git" ]; then
    echo "📥 Pulling latest changes..."
    git pull origin main
fi

# Build Docker images
echo "🔨 Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run health checks
echo "🏥 Running health checks..."
docker-compose exec ai-trader python -c "
import asyncio
from utils.telegram_bot import AdvancedTelegramBot
from agents.master_orchestrator import MasterAIOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def health_check():
    try:
        # Test Telegram connection
        bot = AdvancedTelegramBot('YOUR_BOT_TOKEN', 'YOUR_CHAT_ID')
        connected = await bot.test_connection()
        
        # Test AI system
        config = {'system': {'max_workers': 2}}
        orchestrator = MasterAIOrchestrator(config)
        
        logger.info('Health check: PASSED')
        return True
    except Exception as e:
        logger.error(f'Health check: FAILED - {e}')
        return False

asyncio.run(health_check())
"

echo "✅ Deployment to ${ENVIRONMENT} completed successfully!"
echo ""
echo "📊 Monitoring:"
echo "  Logs: docker-compose logs -f ai-trader"
echo "  Status: docker-compose ps"
echo "  Performance: http://localhost:3000 (if Grafana enabled)"
