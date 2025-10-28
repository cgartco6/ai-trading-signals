"""
Sentiment Intelligence Agent using NLP and Social Media Analysis
"""

import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from textblob import TextBlob
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentIntelligenceAgent:
    """AI agent for market sentiment analysis using NLP"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sentiment_analyzer = None
        self.tokenizer = None
        self.fear_greed_index = 50
        self._initialize_nlp_models()
        
    def _initialize_nlp_models(self):
        """Initialize NLP models for sentiment analysis"""
        try:
            # Initialize transformer-based sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                tokenizer="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
            )
            self.logger.info("NLP models initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not load transformer model: {e}")
            # Fallback to TextBlob
            self.sentiment_analyzer = None
    
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze sentiment for a specific symbol"""
        try:
            # Get sentiment data from multiple sources
            sentiment_scores = await asyncio.gather(
                self._analyze_news_sentiment(symbol),
                self._analyze_social_sentiment(symbol),
                self._analyze_technical_sentiment(symbol),
                return_exceptions=True
            )
            
            # Filter out exceptions
            valid_scores = [score for score in sentiment_scores if not isinstance(score, Exception)]
            
            if not valid_scores:
                return None
            
            # Calculate weighted sentiment score
            sentiment_score = np.mean([score.get('score', 0) for score in valid_scores])
            confidence = np.mean([score.get('confidence', 0) for score in valid_scores])
            
            # Convert sentiment score to trading action
            action, action_confidence = self._sentiment_to_action(sentiment_score, confidence)
            
            if action != 'HOLD':
                return {
                    'action': action,
                    'confidence': action_confidence,
                    'sentiment_score': sentiment_score,
                    'fear_greed_index': self.fear_greed_index,
                    'sources_analyzed': len(valid_scores)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return None
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment for symbol"""
        try:
            # In production, this would fetch from news APIs
            # For now, simulate news sentiment analysis
            news_articles = await self._fetch_news_articles(symbol)
            
            if not news_articles:
                return {'score': 0, 'confidence': 0}
            
            sentiments = []
            confidences = []
            
            for article in news_articles:
                if self.sentiment_analyzer:
                    # Use transformer model
                    result = self.sentiment_analyzer(article['title'] + " " + article['content'])[0]
                    sentiment_score = self._transformers_sentiment_to_score(result)
                    confidence = result['score']
                else:
                    # Use TextBlob fallback
                    blob = TextBlob(article['title'] + " " + article['content'])
                    sentiment_score = blob.sentiment.polarity
                    confidence = abs(sentiment_score)
                
                sentiments.append(sentiment_score)
                confidences.append(confidence)
            
            avg_sentiment = np.mean(sentiments)
            avg_confidence = np.mean(confidences)
            
            return {
                'score': avg_sentiment,
                'confidence': avg_confidence,
                'articles_analyzed': len(news_articles)
            }
            
        except Exception as e:
            self.logger.warning(f"News sentiment analysis failed: {e}")
            return {'score': 0, 'confidence': 0}
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze social media sentiment"""
        try:
            # This would integrate with Twitter API, Reddit API, etc.
            # For now, simulate social sentiment
            social_posts = await self._fetch_social_posts(symbol)
            
            if not social_posts:
                return {'score': 0, 'confidence': 0}
            
            sentiments = []
            for post in social_posts:
                blob = TextBlob(post['content'])
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments)
            confidence = min(0.8, np.std(sentiments) * 10)  # Higher confidence for consistent sentiment
            
            # Update fear and greed index based on social sentiment
            self._update_fear_greed_index(avg_sentiment)
            
            return {
                'score': avg_sentiment,
                'confidence': confidence,
                'posts_analyzed': len(social_posts)
            }
            
        except Exception as e:
            self.logger.warning(f"Social sentiment analysis failed: {e}")
            return {'score': 0, 'confidence': 0}
    
    async def _analyze_technical_sentiment(self, symbol: str) -> Dict:
        """Derive sentiment from technical analysis"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(symbol, '1h', limit=50)
            
            if df is None:
                return {'score': 0, 'confidence': 0}
            
            # Calculate technical indicators that correlate with sentiment
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean()
            volatility = df['close'].pct_change().std()
            
            # Combine factors into sentiment score
            technical_sentiment = (
                np.tanh(price_change * 10) * 0.4 +  # Price momentum
                np.tanh(volume_trend - 1) * 0.3 +   # Volume trend
                np.tanh(volatility * 100) * 0.3     # Volatility
            )
            
            confidence = 0.7  # Technical analysis typically has moderate confidence
            
            return {
                'score': technical_sentiment,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"Technical sentiment analysis failed: {e}")
            return {'score': 0, 'confidence': 0}
    
    def _sentiment_to_action(self, sentiment_score: float, confidence: float) -> tuple:
        """Convert sentiment score to trading action"""
        if sentiment_score > 0.2 and confidence > 0.6:
            return 'BUY', confidence
        elif sentiment_score < -0.2 and confidence > 0.6:
            return 'SELL', confidence
        else:
            return 'HOLD', confidence
    
    def _transformers_sentiment_to_score(self, result: Dict) -> float:
        """Convert transformer sentiment result to numeric score"""
        label = result['label']
        score = result['score']
        
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:
            return 0
    
    def _update_fear_greed_index(self, sentiment_score: float):
        """Update fear and greed index based on sentiment"""
        # Simple update rule - in production, use more sophisticated calculation
        change = sentiment_score * 10
        self.fear_greed_index = max(0, min(100, self.fear_greed_index + change))
    
    async def _fetch_news_articles(self, symbol: str) -> List[Dict]:
        """Fetch news articles for symbol (simulated)"""
        # In production, integrate with NewsAPI, Bloomberg, etc.
        return [
            {
                'title': f"Market analysis for {symbol}",
                'content': f"Recent performance of {symbol} shows positive momentum.",
                'source': 'simulated',
                'published_at': datetime.now()
            }
        ]
    
    async def _fetch_social_posts(self, symbol: str) -> List[Dict]:
        """Fetch social media posts for symbol (simulated)"""
        # In production, integrate with Twitter API, Reddit API, etc.
        return [
            {
                'content': f"Bullish on {symbol} today!",
                'source': 'twitter_simulated',
                'created_at': datetime.now()
            }
        ]
    
    async def analyze_market_regime(self):
        """Analyze market regime based on sentiment"""
        from .master_orchestrator import MarketRegime
        
        if self.fear_greed_index > 75:
            return MarketRegime.TRENDING
        elif self.fear_greed_index < 25:
            return MarketRegime.CRASH
        elif 40 <= self.fear_greed_index <= 60:
            return MarketRegime.RANGING
        else:
            return MarketRegime.VOLATILE
