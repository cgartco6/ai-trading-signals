#!/usr/bin/env python3
"""
Main entry point for AI Trading Signal System
Optimized for Dell i7, 16GB RAM
"""

import asyncio
import argparse
import logging
from utils.config_loader import Config
from utils.logger import setup_logging
from agents.master_orchestrator import MasterAIOrchestrator
from utils.telegram_bot import AdvancedTelegramBot

class AITradingSystem:
    """Main trading system class"""
    
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self.orchestrator = MasterAIOrchestrator(self.config)
        self.telegram_bot = AdvancedTelegramBot(
            self.config.get('telegram.bot_token'),
            self.config.get('telegram.chat_id')
        )
        self.is_running = False
        
    async def start(self):
        """Start the trading system"""
        logging.info("🚀 Starting AI Trading System...")
        self.is_running = True
        
        try:
            # Start continuous signal generation
            await asyncio.gather(
                self._generate_signals_loop(),
                self._monitor_performance(),
                self._handle_health_checks()
            )
        except KeyboardInterrupt:
            logging.info("Shutting down gracefully...")
        finally:
            self.is_running = False
            
    async def _generate_signals_loop(self):
        """Main signal generation loop"""
        while self.is_running:
            try:
                signals = await self.orchestrator.generate_signals()
                await self._process_signals(signals)
                
                # Wait for next interval
                interval = self.config.get('trading.update_interval', 300)
                await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"Signal generation error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def _process_signals(self, signals):
        """Process and send signals"""
        for signal in signals:
            if signal.confidence > self.config.get('trading.confidence_threshold', 0.75):
                await self.telegram_bot.send_signal(signal)
                
    async def _monitor_performance(self):
        """Monitor system performance"""
        while self.is_running:
            # Implement performance monitoring
            await asyncio.sleep(300)  # Every 5 minutes
            
    async def _handle_health_checks(self):
        """Handle health checks"""
        # Implement health check endpoint
        pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Trading Signal System')
    parser.add_argument('--config', type=str, default='environments/local_i7_setup.yaml',
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Create and run system
    system = AITradingSystem(args.config)
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        logging.info("System stopped by user")

if __name__ == "__main__":
    main()
