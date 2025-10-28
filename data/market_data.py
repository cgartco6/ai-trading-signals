"""
Market Data Fetcher for Forex and Crypto
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time

class MarketDataFetcher:
    """Fetches market data from various sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
        # API endpoints
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.alphavantage_base_url = "https://www.alphavantage.co/query"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_symbol_data(self, symbol: str, interval: str = '1h', 
                            limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol"""
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Determine data source based on symbol
            if '/USDT' in symbol or 'USDT' in symbol:
                # Crypto pair
                df = await self._get_binance_data(symbol, interval, limit)
            else:
                # Forex pair (simulated for now)
                df = await self._get_forex_data(symbol, interval, limit)
            
            if df is not None:
                # Cache the data
                self.cache[cache_key] = (df, time.time())
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    async def _get_binance_data(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Binance API"""
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            # Convert symbol to Binance format
            binance_symbol = symbol.replace('/', '').replace('USDT', 'USDT')
            
            url = f"{self.binance_base_url}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': self._convert_interval(interval),
                'limit': limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert to proper data types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    return df[['open', 'high', 'low', 'close', 'volume']]
                else:
                    self.logger.error(f"Binance API error: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Binance data fetch failed: {e}")
            return None
    
    async def _get_forex_data(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch Forex data (simulated for now)"""
        try:
            # In production, this would use a Forex API like OANDA, Alpha Vantage, etc.
            # For now, generate simulated data
            
            # Remove any forward slashes for filename
            safe_symbol = symbol.replace('/', '_')
            
            # Try to load from cached file first
            try:
                df = pd.read_csv(f'data/cache/{safe_symbol}_{interval}.csv', index_col='timestamp', parse_dates=True)
                if len(df) >= limit:
                    return df.tail(limit)
            except FileNotFoundError:
                pass
            
            # Generate simulated data
            np.random.seed(hash(symbol) % 10000)
            
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=self._get_pandas_freq(interval))
            
            # Start with random price
            base_price = 100 if 'JPY' not in symbol else 1.0
            prices = [base_price]
            
            for i in range(1, limit):
                # Random walk with drift
                returns = np.random.normal(0.0001, 0.01)  # 1% daily volatility
                new_price = prices[-1] * (1 + returns)
                prices.append(new_price)
            
            # Create OHLCV data
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': np.random.lognormal(10, 1, limit)
            }, index=dates)
            
            # Ensure high >= open, high >= close, low <= open, low <= close
            df['high'] = df[['open', 'close', 'high']].max(axis=1)
            df['low'] = df[['open', 'close', 'low']].min(axis=1)
            
            # Save to cache
            df.to_csv(f'data/cache/{safe_symbol}_{interval}.csv')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Forex data generation failed: {e}")
            return None
    
    def _convert_interval(self, interval: str) -> str:
        """Convert trading interval to exchange format"""
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }
        return interval_map.get(interval, '1h')
    
    def _get_pandas_freq(self, interval: str) -> str:
        """Convert interval to pandas frequency"""
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D',
            '1w': '1W'
        }
        return freq_map.get(interval, '1h')
    
    async def get_multiple_symbols(self, symbols: List[str], interval: str = '1h', 
                                 limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols concurrently"""
        tasks = []
        for symbol in symbols:
            task = self.get_symbol_data(symbol, interval, limit)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        symbol_data = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception) and result is not None:
                symbol_data[symbol] = result
        
        return symbol_data
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()

# Example usage
async def main():
    async with MarketDataFetcher({}) as fetcher:
        # Get Bitcoin data
        btc_data = await fetcher.get_symbol_data('BTC/USDT', '1h', 100)
        print(f"BTC Data shape: {btc_data.shape if btc_data is not None else 'No data'}")
        
        # Get Forex data
        eur_usd_data = await fetcher.get_symbol_data('EUR/USD', '1h', 100)
        print(f"EUR/USD Data shape: {eur_usd_data.shape if eur_usd_data is not None else 'No data'}")

if __name__ == "__main__":
    asyncio.run(main())
