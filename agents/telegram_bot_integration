class TelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_signal(self, signal):
        """Send trading signal to Telegram channel"""
        try:
            if signal['signal'] == 'HOLD':
                return
                
            message = self.format_signal_message(signal)
            
            url = f"{self.base_url}/sendMessage"
            params = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=params)
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")
            return False
    
    def format_signal_message(self, signal):
        """Format the signal into a nice Telegram message"""
        symbol = signal['symbol']
        action = signal['signal']
        price = signal['price']
        confidence = signal['confidence']
        
        if action == 'BUY':
            emoji = "🟢"
            action_text = "BUY"
        else:
            emoji = "🔴"
            action_text = "SELL"
            
        message = f"""
{emoji} <b>TRADING SIGNAL</b> {emoji}

📈 <b>Symbol:</b> {symbol}
🎯 <b>Action:</b> {action_text}
💰 <b>Current Price:</b> ${price:.4f}
📊 <b>Confidence:</b> {confidence:.1f}%

<b>Technical Indicators:</b>
"""
        
        for indicator, status in signal['indicators'].items():
            if status in ['BULLISH', 'OVERSOLD']:
                message += f"✅ {indicator.upper()}: {status}\n"
            else:
                message += f"⚠️ {indicator.upper()}: {status}\n"
                
        message += f"\n⏰ <i>Generated: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</i>"
        message += f"\n\n⚠️ <i>Always do your own research and use proper risk management!</i>"
        
        return message
