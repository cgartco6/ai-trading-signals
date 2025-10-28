# config.py
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# main.py
if __name__ == "__main__":
    # Initialize the bot
    bot = TradingSignalBot(
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    
    # Run once
    # signals = bot.run_analysis()
    
    # Run continuously
    bot.start_monitoring(interval_minutes=60)  # Check every hour
