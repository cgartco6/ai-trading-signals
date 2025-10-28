"""
Advanced Telegram Bot for Trading Signals
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime
from ..agents.master_orchestrator import TradingSignal

class AdvancedTelegramBot:
    """Enhanced Telegram bot with rich formatting and interactive features"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.message_history = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_signal(self, signal: TradingSignal) -> bool:
        """Send trading signal to Telegram"""
        try:
            message = self._format_signal_message(signal)
            success = await self._send_message(message)
            
            if success:
                self.logger.info(f"Signal sent successfully: {signal.signal_id}")
                self._store_message_history(signal, message)
            else:
                self.logger.error(f"Failed to send signal: {signal.signal_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending signal: {e}")
            return False
    
    async def send_batch_signals(self, signals: List[TradingSignal]) -> bool:
        """Send multiple signals in a batch"""
        if not signals:
            return True
        
        try:
            message = self._format_batch_signals_message(signals)
            success = await self._send_message(message)
            
            if success:
                self.logger.info(f"Batch of {len(signals)} signals sent successfully")
            else:
                self.logger.error(f"Failed to send batch of {len(signals)} signals")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending batch signals: {e}")
            return False
    
    async def send_alert(self, title: str, message: str, level: str = "info") -> bool:
        """Send alert message with different levels"""
        try:
            formatted_message = self._format_alert_message(title, message, level)
            success = await self._send_message(formatted_message)
            
            if success:
                self.logger.info(f"Alert sent: {title}")
            else:
                self.logger.error(f"Failed to send alert: {title}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            return False
    
    async def send_performance_report(self, performance_data: Dict) -> bool:
        """Send performance report"""
        try:
            message = self._format_performance_message(performance_data)
            success = await self._send_message(message)
            
            if success:
                self.logger.info("Performance report sent successfully")
            else:
                self.logger.error("Failed to send performance report")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending performance report: {e}")
            return False
    
    def _format_signal_message(self, signal: TradingSignal) -> str:
        """Format trading signal into Telegram message"""
        # Emojis for different actions
        emoji_map = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '⚪'
        }
        
        action_emoji = emoji_map.get(signal.action, '⚪')
        
        # Confidence indicator
        if signal.confidence >= 0.9:
            confidence_emoji = '🎯'
        elif signal.confidence >= 0.8:
            confidence_emoji = '📈'
        elif signal.confidence >= 0.7:
            confidence_emoji = '📊'
        else:
            confidence_emoji = '📉'
        
        # Market regime emoji
        regime_emoji = {
            'TRENDING': '🚀',
            'RANGING': '➰',
            'VOLATILE': '🌊',
            'CRASH': '💥',
            'RECOVERY': '🔄'
        }.get(signal.market_regime.value, '📈')
        
        message = f"""
{action_emoji} <b>AI TRADING SIGNAL</b> {action_emoji}

<b>Symbol:</b> {signal.symbol}
<b>Action:</b> <code>{signal.action}</code>
<b>Confidence:</b> {confidence_emoji} {signal.confidence:.1%}
<b>Timeframe:</b> {signal.timeframe}
<b>Market Regime:</b> {regime_emoji} {signal.market_regime.value}

<b>Price Levels:</b>
├ Entry: <code>${signal.entry_price:.4f}</code>
├ Stop Loss: <code>${signal.stop_loss:.4f}</code>
└ Targets: {', '.join(f'<code>${t:.4f}</code>' for t in signal.targets)}

<b>AI Agents Consensus:</b>
"""
        
        # Add agent breakdown
        for agent_name, prediction in signal.agent_breakdown.items():
            agent_action = prediction.get('action', 'HOLD')
            agent_confidence = prediction.get('confidence', 0)
            agent_emoji = '✅' if agent_action == signal.action else '⚪'
            
            message += f"{agent_emoji} {agent_name}: {agent_action} ({agent_confidence:.1%})\n"
        
        message += f"\n<b>Signal ID:</b> <code>{signal.signal_id}</code>"
        message += f"\n<b>Generated:</b> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        # Risk disclaimer
        message += f"\n\n⚠️ <i>Always do your own research and use proper risk management!</i>"
        
        return message
    
    def _format_batch_signals_message(self, signals: List[TradingSignal]) -> str:
        """Format multiple signals into a single message"""
        if len(signals) == 1:
            return self._format_signal_message(signals[0])
        
        message = f"📊 <b>BATCH TRADING SIGNALS</b> 📊\n\n"
        message += f"<b>Total Signals:</b> {len(signals)}\n\n"
        
        for i, signal in enumerate(signals, 1):
            action_emoji = '🟢' if signal.action == 'BUY' else '🔴'
            message += f"{i}. {action_emoji} <b>{signal.symbol}</b> - {signal.action} ({signal.confidence:.1%})\n"
        
        message += f"\n⏰ <i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        message += f"\n\n⚠️ <i>Review individual signals for detailed analysis</i>"
        
        return message
    
    def _format_alert_message(self, title: str, message: str, level: str) -> str:
        """Format alert message"""
        level_emojis = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '🚨',
            'success': '✅'
        }
        
        emoji = level_emojis.get(level, 'ℹ️')
        
        return f"""
{emoji} <b>{title}</b> {emoji}

{message}

<i>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
    
    def _format_performance_message(self, performance_data: Dict) -> str:
        """Format performance report"""
        message = "📈 <b>TRADING PERFORMANCE REPORT</b> 📈\n\n"
        
        # Overall performance
        if 'overall' in performance_data:
            overall = performance_data['overall']
            message += f"<b>Overall Performance:</b>\n"
            message += f"├ Total Signals: {overall.get('total_signals', 0)}\n"
            message += f"├ Successful: {overall.get('successful_signals', 0)}\n"
            message += f"├ Accuracy: {overall.get('accuracy', 0):.1%}\n"
            message += f"└ Avg Confidence: {overall.get('avg_confidence', 0):.1%}\n\n"
        
        # Agent performance
        if 'agents' in performance_data:
            message += "<b>Agent Performance:</b>\n"
            for agent, stats in performance_data['agents'].items():
                message += f"├ {agent}: {stats.get('accuracy', 0):.1%} ({stats.get('total_signals', 0)} signals)\n"
            message += "\n"
        
        # Recent signals
        if 'recent_signals' in performance_data:
            recent = performance_data['recent_signals']
            message += f"<b>Recent Signals (Last 24h):</b> {len(recent)}\n"
        
        message += f"\n<i>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        
        return message
    
    async def _send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message to Telegram"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Telegram API error: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def _store_message_history(self, signal: TradingSignal, message: str):
        """Store message in history"""
        history_entry = {
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'action': signal.action,
            'confidence': signal.confidence,
            'timestamp': datetime.now(),
            'message': message
        }
        
        self.message_history.append(history_entry)
        
        # Keep only last 100 messages
        if len(self.message_history) > 100:
            self.message_history = self.message_history[-100:]
    
    async def get_bot_info(self) -> Optional[Dict]:
        """Get bot information"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        try:
            url = f"{self.base_url}/getMe"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('result')
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error getting bot info: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Test Telegram connection"""
        bot_info = await self.get_bot_info()
        if bot_info and bot_info.get('is_bot'):
            self.logger.info(f"Telegram bot connected: {bot_info.get('username')}")
            return True
        else:
            self.logger.error("Failed to connect to Telegram bot")
            return False

# Example usage
async def main():
    # Test the Telegram bot
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    
    async with AdvancedTelegramBot(bot_token, chat_id) as bot:
        # Test connection
        connected = await bot.test_connection()
        print(f"Bot connected: {connected}")
        
        if connected:
            # Test alert
            await bot.send_alert(
                title="System Started",
                message="AI Trading System has started successfully.",
                level="success"
            )

if __name__ == "__main__":
    asyncio.run(main())
