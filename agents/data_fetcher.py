class DataFetcher:
    def __init__(self):
        self.crypto_base_url = "https://api.binance.com/api/v3"
        self.forex_base_url = "https://api.twelvedata.com"  # You'll need an API key
        
    def fetch_crypto_data(self, symbol, interval='1h', limit=100):
        """Fetch cryptocurrency data from Binance"""
        try:
            url = f"{self.crypto_base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            return df
            
        except Exception as e:
            logging.error(f"Error fetching crypto data for {symbol}: {e}")
            return None
    
    def fetch_forex_data(self, symbol, interval='1h', limit=100):
        """Fetch Forex data (you'll need to implement with your preferred API)"""
        # Example using Twelve Data (requires API key)
        try:
            url = f"{self.forex_base_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': limit,
                'apikey': 'YOUR_API_KEY'  # Replace with your API key
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data['values'])
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
                
            return df
            
        except Exception as e:
            logging.error(f"Error fetching Forex data for {symbol}: {e}")
            return None
