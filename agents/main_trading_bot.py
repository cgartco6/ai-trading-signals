class TradingSignalBot:
    def __init__(self, telegram_bot_token, telegram_chat_id):
        self.signal_generator = TradingSignalGenerator()
        self.data_fetcher = DataFetcher()
        self.telegram_bot = TelegramBot(telegram_bot_token, telegram_chat_id)
        
        # Define symbols to monitor
        self.crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        self.forex_symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
        
        # Track recent signals to avoid spam
        self.recent_signals = {}
        
    def run_analysis(self):
        """Run analysis on all symbols"""
        all_signals = []
        
        # Analyze cryptocurrencies
        for symbol in self.crypto_symbols:
            print(f"Analyzing {symbol}...")
            df = self.data_fetcher.fetch_crypto_data(symbol)
            if df is not None:
                signal = self.signal_generator.generate_signal(symbol, df, 'crypto')
                if signal and self.should_send_signal(signal):
                    all_signals.append(signal)
            
            time.sleep(1)  # Rate limiting
            
        # Analyze Forex pairs (implement based on your data source)
        # for symbol in self.forex_symbols:
        #     df = self.data_fetcher.fetch_forex_data(symbol)
        #     if df is not None:
        #         signal = self.signal_generator.generate_signal(symbol, df, 'forex')
        #         if signal and self.should_send_signal(signal):
        #             all_signals.append(signal)
        #     time.sleep(1)
        
        # Send signals
        for signal in all_signals:
            success = self.telegram_bot.send_signal(signal)
            if success:
                self.recent_signals[signal['symbol']] = signal['timestamp']
                print(f"Signal sent for {signal['symbol']}")
            time.sleep(1)
            
        return all_signals
    
    def should_send_signal(self, signal):
        """Check if we should send this signal (avoid spamming)"""
        symbol = signal['symbol']
        current_time = signal['timestamp']
        
        if symbol in self.recent_signals:
            last_signal_time = self.recent_signals[symbol]
            time_diff = (current_time - last_signal_time).total_seconds() / 3600  # hours
            
            # Don't send signals for the same symbol more than once every 4 hours
            if time_diff < 4:
                return False
                
        return True
    
    def start_monitoring(self, interval_minutes=60):
        """Start continuous monitoring"""
        print(f"Starting monitoring bot. Interval: {interval_minutes} minutes")
        
        while True:
            try:
                print(f"\n--- Running analysis at {datetime.now()} ---")
                signals = self.run_analysis()
                print(f"Generated {len(signals)} signals")
                
                print(f"Waiting {interval_minutes} minutes until next analysis...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("Bot stopped by user")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
