import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import threading
import time

class CryptoDataManager:
    """Manages cryptocurrency data and analysis with configurable top crypto system"""
    
    def __init__(self):
        # Data cache system for crypto
        self._crypto_cache = {}
        self._crypto_cache_timestamp = None
        self._crypto_cache_lock = threading.Lock()
        self._crypto_cache_expiry_hours = 4  # Cache expires after 4 hours
        self._is_crypto_cache_warming = False
        
        # Comprehensive list of popular cryptocurrencies available on Yahoo Finance
        self.ALL_CRYPTO_SYMBOLS = {
            # Major Store of Value / Digital Gold
            'BTC-USD': {'name': 'Bitcoin', 'category': 'Crypto'},
            
            # Smart Contract Platforms
            'ETH-USD': {'name': 'Ethereum', 'category': 'Crypto'},
            'ADA-USD': {'name': 'Cardano', 'category': 'Crypto'},
            'SOL-USD': {'name': 'Solana', 'category': 'Crypto'},
            'DOT-USD': {'name': 'Polkadot', 'category': 'Crypto'},
            'AVAX-USD': {'name': 'Avalanche', 'category': 'Crypto'},
            'ALGO-USD': {'name': 'Algorand', 'category': 'Crypto'},
            'ATOM-USD': {'name': 'Cosmos', 'category': 'Crypto'},
            'NEAR-USD': {'name': 'NEAR Protocol', 'category': 'Crypto'},
            
            # Exchange Tokens
            'BNB-USD': {'name': 'BNB', 'category': 'Crypto'},
            'CRO-USD': {'name': 'Cronos', 'category': 'Crypto'},
            'FTT-USD': {'name': 'FTX Token', 'category': 'Crypto'},
            'HT-USD': {'name': 'Huobi Token', 'category': 'Crypto'},
            
            # Payment/Transfer
            'XRP-USD': {'name': 'XRP', 'category': 'Crypto'},
            'LTC-USD': {'name': 'Litecoin', 'category': 'Crypto'},
            'BCH-USD': {'name': 'Bitcoin Cash', 'category': 'Crypto'},
            'XLM-USD': {'name': 'Stellar', 'category': 'Crypto'},
            'XMR-USD': {'name': 'Monero', 'category': 'Crypto'},
            'DASH-USD': {'name': 'Dash', 'category': 'Crypto'},
            
            # DeFi Tokens
            'UNI-USD': {'name': 'Uniswap', 'category': 'Crypto'},
            'LINK-USD': {'name': 'Chainlink', 'category': 'Crypto'},
            'AAVE-USD': {'name': 'Aave', 'category': 'Crypto'},
            'COMP-USD': {'name': 'Compound', 'category': 'Crypto'},
            'SUSHI-USD': {'name': 'SushiSwap', 'category': 'Crypto'},
            'CRV-USD': {'name': 'Curve DAO', 'category': 'Crypto'},
            
            # Layer 2 Solutions
            'MATIC-USD': {'name': 'Polygon', 'category': 'Crypto'},
            'LRC-USD': {'name': 'Loopring', 'category': 'Crypto'},
            'IMX-USD': {'name': 'Immutable X', 'category': 'Crypto'},
            
            # Meme Coins
            'DOGE-USD': {'name': 'Dogecoin', 'category': 'Crypto'},
            'SHIB-USD': {'name': 'Shiba Inu', 'category': 'Crypto'},
            
            # Gaming/NFT
            'MANA-USD': {'name': 'Decentraland', 'category': 'Crypto'},
            'SAND-USD': {'name': 'The Sandbox', 'category': 'Crypto'},
            'AXS-USD': {'name': 'Axie Infinity', 'category': 'Crypto'},
            'ENJ-USD': {'name': 'Enjin Coin', 'category': 'Crypto'},
            
            # Infrastructure
            'LUNA-USD': {'name': 'Terra Luna', 'category': 'Crypto'},
            'FIL-USD': {'name': 'Filecoin', 'category': 'Crypto'},
            'AR-USD': {'name': 'Arweave', 'category': 'Crypto'},
            
            # Stablecoins (for reference)
            'USDT-USD': {'name': 'Tether', 'category': 'Crypto'},
            'USDC-USD': {'name': 'USD Coin', 'category': 'Crypto'},
            'BUSD-USD': {'name': 'Binance USD', 'category': 'Crypto'},
            
            # Other Notable
            'TRX-USD': {'name': 'TRON', 'category': 'Crypto'},
            'VET-USD': {'name': 'VeChain', 'category': 'Crypto'},
            'THETA-USD': {'name': 'Theta Network', 'category': 'Crypto'},
            'ICP-USD': {'name': 'Internet Computer', 'category': 'Crypto'},
            'FTM-USD': {'name': 'Fantom', 'category': 'Crypto'},
            'HBAR-USD': {'name': 'Hedera', 'category': 'Crypto'},
            'EOS-USD': {'name': 'EOS', 'category': 'Crypto'},
            'XTZ-USD': {'name': 'Tezos', 'category': 'Crypto'}
        }
        
        # Default top 10 cryptocurrencies by market cap and popularity
        self.DEFAULT_TOP_CRYPTO = [
            'BTC-USD',   # Bitcoin
            'ETH-USD',   # Ethereum
            'BNB-USD',   # BNB
            'XRP-USD',   # XRP
            'ADA-USD',   # Cardano
            'DOGE-USD',  # Dogecoin
            'SOL-USD',   # Solana
            'TRX-USD',   # TRON
            'DOT-USD',   # Polkadot
            'MATIC-USD'  # Polygon
        ]
        
        # Current active top crypto (configurable)
        self._top_crypto = self.DEFAULT_TOP_CRYPTO.copy()
        logging.info(f"Loaded {len(self.ALL_CRYPTO_SYMBOLS)} crypto symbols, tracking top {len(self._top_crypto)}")
        
        # Start crypto cache warming in background
        self._warm_crypto_cache_async()
        
    def get_all_crypto_symbols(self) -> List[str]:
        """Get all cryptocurrency symbols"""
        return list(self.ALL_CRYPTO_SYMBOLS.keys())
    
    def get_top_crypto(self) -> List[str]:
        """Get current top crypto list"""
        return self._top_crypto.copy()
    
    def set_top_crypto(self, symbols: List[str]) -> bool:
        """Set new top crypto list (must be valid crypto symbols)"""
        try:
            # Validate all symbols are in our crypto list
            invalid_symbols = [s for s in symbols if s not in self.ALL_CRYPTO_SYMBOLS]
            if invalid_symbols:
                logging.warning(f"Invalid crypto symbols: {invalid_symbols}")
                return False
            
            self._top_crypto = symbols
            logging.info(f"Updated top crypto list to {len(symbols)} cryptocurrencies")
            return True
        except Exception as e:
            logging.error(f"Error setting top crypto: {e}")
            return False
    
    def add_to_top_crypto(self, symbol: str) -> bool:
        """Add a crypto to the top crypto list"""
        if symbol in self.ALL_CRYPTO_SYMBOLS and symbol not in self._top_crypto:
            self._top_crypto.append(symbol)
            logging.info(f"Added {symbol} to top crypto")
            return True
        return False
    
    def remove_from_top_crypto(self, symbol: str) -> bool:
        """Remove a crypto from the top crypto list"""
        if symbol in self._top_crypto:
            self._top_crypto.remove(symbol)
            logging.info(f"Removed {symbol} from top crypto")
            return True
        return False
    
    def reset_to_default_top_crypto(self):
        """Reset to default top 10 crypto"""
        self._top_crypto = self.DEFAULT_TOP_CRYPTO.copy()
        logging.info("Reset to default top 10 crypto")
        # Warm cache with new crypto list
        self._warm_crypto_cache_async()
    
    def _warm_crypto_cache_async(self):
        """Start crypto cache warming in background thread"""
        if not self._is_crypto_cache_warming:
            thread = threading.Thread(target=self._warm_crypto_cache, daemon=True)
            thread.start()
    
    def _warm_crypto_cache(self):
        """Warm the cache with crypto data at startup"""
        try:
            self._is_crypto_cache_warming = True
            logging.info("Starting crypto cache warming...")
            
            # Warm cache for top crypto
            for symbol in self._top_crypto:
                try:
                    self._fetch_and_cache_crypto_data(symbol)
                    time.sleep(0.1)  # Small delay to avoid hitting rate limits
                except Exception as e:
                    logging.error(f"Error caching crypto data for {symbol}: {e}")
                    continue
            
            self._crypto_cache_timestamp = datetime.now()
            logging.info(f"Crypto cache warming completed for {len(self._top_crypto)} cryptocurrencies")
            
        except Exception as e:
            logging.error(f"Error during crypto cache warming: {e}")
        finally:
            self._is_crypto_cache_warming = False
    
    def _fetch_and_cache_crypto_data(self, symbol):
        """Fetch and cache crypto data for a symbol"""
        try:
            crypto = yf.Ticker(symbol)
            hist = crypto.history(period="5y")
            
            if not hist.empty:
                with self._crypto_cache_lock:
                    self._crypto_cache[symbol] = {
                        'history': hist,
                        'timestamp': datetime.now()
                    }
                logging.debug(f"Cached crypto data for {symbol}")
        except Exception as e:
            logging.error(f"Error fetching crypto data for {symbol}: {e}")
    
    def _get_cached_crypto_data(self, symbol):
        """Get cached crypto data for a symbol, or fetch if not available"""
        with self._crypto_cache_lock:
            # Check if we have cached data
            if symbol in self._crypto_cache:
                cached_data = self._crypto_cache[symbol]
                # Check if cache is still valid (within expiry time)
                if self._crypto_cache_timestamp and (datetime.now() - self._crypto_cache_timestamp).seconds < (self._crypto_cache_expiry_hours * 3600):
                    return cached_data['history']
            
            # Cache miss or expired - fetch new data
            try:
                crypto = yf.Ticker(symbol)
                hist = crypto.history(period="5y")
                
                if not hist.empty:
                    self._crypto_cache[symbol] = {
                        'history': hist,
                        'timestamp': datetime.now()
                    }
                    return hist
                else:
                    return None
            except Exception as e:
                logging.error(f"Error fetching crypto data for {symbol}: {e}")
                return None
    
    def get_crypto_info(self, symbol: str) -> Dict:
        """Get detailed cryptocurrency information"""
        crypto_data = self.ALL_CRYPTO_SYMBOLS.get(symbol, {})
        return {
            'name': crypto_data.get('name', symbol),
            'category': crypto_data.get('category', 'Cryptocurrency'),
            'symbol': symbol
        }
    
    def get_all_categories(self) -> List[str]:
        """Get list of all crypto categories"""
        categories = set(crypto['category'] for crypto in self.ALL_CRYPTO_SYMBOLS.values())
        return sorted(list(categories))
    
    def get_category_summary(self) -> Dict[str, int]:
        """Get count of cryptocurrencies per category"""
        category_counts = {}
        for crypto in self.ALL_CRYPTO_SYMBOLS.values():
            category = crypto['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def get_cryptos_by_category(self, category: str) -> List[str]:
        """Get all crypto symbols in a specific category"""
        return [symbol for symbol, data in self.ALL_CRYPTO_SYMBOLS.items() 
                if data['category'] == category]
    
    def check_crypto_support(self, symbol: str, threshold: float = 0.03) -> Dict:
        """Check if cryptocurrency is near historical support levels (1M, 6M, 1Y, 5Y)"""
        try:
            # Use cached crypto data instead of fetching every time
            hist = self._get_cached_crypto_data(symbol)
            
            if hist is None or hist.empty or len(hist) < 30:
                return {'symbol': symbol, 'price': None, 'zones': 'Insufficient data', 'error': True}
            
            closes = hist["Close"]
            current_price = closes.iloc[-1]
            
            # Calculate support levels based on historical lows in different timeframes
            support_1m = self._calculate_crypto_period_support(closes, 30)  # ~1 month
            support_6m = self._calculate_crypto_period_support(closes, 180)  # ~6 months  
            support_1y = self._calculate_crypto_period_support(closes, 365)  # ~1 year
            support_5y = self._calculate_crypto_period_support(closes, len(closes))  # All available data
            
            zones = []
            support_prices = []
            
            # Check if current price is near each support level
            # IMPORTANT: Price must be ABOVE support level to be considered "near support"
            support_levels = [
                (support_1m, '1M Support'),
                (support_6m, '6M Support'), 
                (support_1y, '1Y Support'),
                (support_5y, '5Y Support')
            ]
            
            for support_price, period in support_levels:
                if support_price and current_price > support_price:  # Price must be above support
                    # Calculate how close price is to support (as percentage above support)
                    distance_from_support = (current_price - support_price) / support_price
                    if distance_from_support <= threshold:  # Within threshold distance above support
                        zones.append(period)
                        support_prices.append(round(support_price, 4))
            
            support_prices_str = ', '.join([f"${price}" for price in support_prices]) if support_prices else '—'
            
            # Generate TradingView link if crypto is near support
            tradingview_link = None
            if zones:  # Only generate link if actually near support
                tradingview_symbol = self._get_tradingview_crypto_symbol(symbol)
                tradingview_link = f"https://www.tradingview.com/chart/?symbol={tradingview_symbol}"
            
            # Get crypto info
            crypto_info = self.get_crypto_info(symbol)
            
            return {
                'ticker': symbol,
                'symbol': symbol,
                'price': round(current_price, 4),
                'zones': ', '.join(zones) or '—',
                'support_prices': support_prices_str,
                'support_levels': support_prices,  # Array for programmatic access
                'support_1m': round(support_1m, 4) if support_1m else None,
                'support_6m': round(support_6m, 4) if support_6m else None,
                'support_1y': round(support_1y, 4) if support_1y else None,
                'support_5y': round(support_5y, 4) if support_5y else None,
                'volume': int(hist["Volume"].iloc[-1]) if not hist["Volume"].empty else None,
                'sector': crypto_info['category'],
                'company_name': crypto_info['name'],
                'industry': 'Cryptocurrency',
                'is_halal': True,  # All crypto assumed halal
                'tradingview_link': tradingview_link,
                'error': False,
                'asset_type': 'crypto'
            }
        except Exception as e:
            logging.error(f"Error checking support for {symbol}: {e}")
            crypto_info = self.get_crypto_info(symbol)
            return {
                'ticker': symbol, 
                'symbol': symbol,
                'price': None, 
                'zones': 'Error fetching data', 
                'sector': crypto_info['category'],
                'company_name': crypto_info['name'],
                'industry': 'Cryptocurrency',
                'error': True,
                'asset_type': 'crypto'
            }
    
    def _get_tradingview_crypto_symbol(self, symbol: str) -> str:
        """Get the appropriate TradingView symbol format for a cryptocurrency"""
        
        # Map Yahoo Finance crypto symbols to TradingView format
        crypto_mapping = {
            'BTC-USD': 'BINANCE:BTCUSDT',
            'ETH-USD': 'BINANCE:ETHUSDT',
            'BNB-USD': 'BINANCE:BNBUSDT',
            'XRP-USD': 'BINANCE:XRPUSDT',
            'ADA-USD': 'BINANCE:ADAUSDT',
            'DOGE-USD': 'BINANCE:DOGEUSDT',
            'SOL-USD': 'BINANCE:SOLUSDT',
            'TRX-USD': 'BINANCE:TRXUSDT',
            'DOT-USD': 'BINANCE:DOTUSDT',
            'MATIC-USD': 'BINANCE:MATICUSDT',
            'LTC-USD': 'BINANCE:LTCUSDT',
            'BCH-USD': 'BINANCE:BCHUSDT',
            'AVAX-USD': 'BINANCE:AVAXUSDT',
            'ALGO-USD': 'BINANCE:ALGOUSDT',
            'ATOM-USD': 'BINANCE:ATOMUSDT',
            'NEAR-USD': 'BINANCE:NEARUSDT',
            'UNI-USD': 'BINANCE:UNIUSDT',
            'LINK-USD': 'BINANCE:LINKUSDT',
            'AAVE-USD': 'BINANCE:AAVEUSDT',
            'COMP-USD': 'BINANCE:COMPUSDT',
            'SUSHI-USD': 'BINANCE:SUSHIUSDT',
            'CRV-USD': 'BINANCE:CRVUSDT',
            'LRC-USD': 'BINANCE:LRCUSDT',
            'SHIB-USD': 'BINANCE:SHIBUSDT',
            'MANA-USD': 'BINANCE:MANAUSDT',
            'SAND-USD': 'BINANCE:SANDUSDT',
            'AXS-USD': 'BINANCE:AXSUSDT',
            'ENJ-USD': 'BINANCE:ENJUSDT',
            'FIL-USD': 'BINANCE:FILUSDT',
            'VET-USD': 'BINANCE:VETUSDT',
            'THETA-USD': 'BINANCE:THETAUSDT',
            'ICP-USD': 'BINANCE:ICPUSDT',
            'FTM-USD': 'BINANCE:FTMUSDT',
            'HBAR-USD': 'BINANCE:HBARUSDT',
            'EOS-USD': 'BINANCE:EOSUSDT',
            'XTZ-USD': 'BINANCE:XTZUSDT',
            'XLM-USD': 'BINANCE:XLMUSDT',
            'XMR-USD': 'KRAKEN:XMRUSD',  # Monero often on Kraken
            'DASH-USD': 'BINANCE:DASHUSDT',
            'USDT-USD': 'BINANCE:USDTUSD',
            'USDC-USD': 'BINANCE:USDCUSDT'
        }
        
        # Return mapped symbol or construct default
        if symbol in crypto_mapping:
            return crypto_mapping[symbol]
        else:
            # Fallback: convert BTC-USD to BTCUSDT format
            base_symbol = symbol.replace('-USD', '').replace('-', '')
            return f"BINANCE:{base_symbol}USDT"
    
    def _calculate_crypto_period_support(self, closes, lookback_days):
        """Calculate support level for a given period using lowest low and key support zones"""
        try:
            if len(closes) < lookback_days:
                lookback_days = len(closes)
            
            period_data = closes.tail(lookback_days)
            
            # Find the lowest low in the period
            period_low = period_data.min()
            
            # Also consider areas where price has bounced multiple times (support zones)
            # Calculate support as the 15th percentile of prices in the period (crypto is more volatile)
            support_zone = period_data.quantile(0.15)
            
            # Return the higher of the two (more conservative support)
            return max(period_low, support_zone)
            
        except Exception:
            return None
    
    def check_crypto_resistance(self, symbol: str, threshold: float = 0.03) -> Dict:
        """Check if cryptocurrency is near historical resistance levels (1M, 6M, 1Y, 5Y)"""
        try:
            # Use cached crypto data instead of fetching every time
            hist = self._get_cached_crypto_data(symbol)
            
            if hist is None or hist.empty or len(hist) < 30:
                return {'symbol': symbol, 'price': None, 'zones': 'Insufficient data', 'error': True}
            
            closes = hist["Close"]
            current_price = closes.iloc[-1]
            
            # Calculate resistance levels based on historical highs in different timeframes
            resistance_1m = self._calculate_crypto_period_resistance(closes, 30)  # ~1 month
            resistance_6m = self._calculate_crypto_period_resistance(closes, 180)  # ~6 months  
            resistance_1y = self._calculate_crypto_period_resistance(closes, 365)  # ~1 year
            resistance_5y = self._calculate_crypto_period_resistance(closes, len(closes))  # All available data
            
            zones = []
            resistance_prices = []
            
            # Check if current price is near each resistance level
            resistance_levels = [
                (resistance_1m, '1M Resistance'),
                (resistance_6m, '6M Resistance'), 
                (resistance_1y, '1Y Resistance'),
                (resistance_5y, '5Y Resistance')
            ]
            
            for resistance_price, period in resistance_levels:
                if resistance_price and current_price < resistance_price:  # Price must be below resistance
                    distance_from_resistance = (resistance_price - current_price) / resistance_price
                    if distance_from_resistance <= threshold:
                        zones.append(period)
                        resistance_prices.append(round(resistance_price, 4))
            
            resistance_prices_str = ', '.join([f"${price}" for price in resistance_prices]) if resistance_prices else '—'
            
            # Generate TradingView link if crypto is near resistance
            tradingview_link = None
            if zones:
                tradingview_symbol = self._get_tradingview_crypto_symbol(symbol)
                tradingview_link = f"https://www.tradingview.com/chart/?symbol={tradingview_symbol}"
            
            # Get crypto info
            crypto_info = self.get_crypto_info(symbol)
            
            return {
                'ticker': symbol,
                'symbol': symbol,
                'price': round(current_price, 4),
                'zones': ', '.join(zones) or '—',
                'resistance_prices': resistance_prices_str,
                'resistance_levels': resistance_prices,
                'resistance_1m': round(resistance_1m, 4) if resistance_1m else None,
                'resistance_6m': round(resistance_6m, 4) if resistance_6m else None,
                'resistance_1y': round(resistance_1y, 4) if resistance_1y else None,
                'resistance_5y': round(resistance_5y, 4) if resistance_5y else None,
                'volume': int(hist["Volume"].iloc[-1]) if not hist["Volume"].empty else None,
                'sector': crypto_info['category'],
                'company_name': crypto_info['name'],
                'industry': 'Cryptocurrency',
                'is_halal': True,
                'tradingview_link': tradingview_link,
                'error': False,
                'asset_type': 'crypto'
            }
        except Exception as e:
            logging.error(f"Error checking resistance for {symbol}: {e}")
            crypto_info = self.get_crypto_info(symbol)
            return {
                'ticker': symbol, 
                'symbol': symbol,
                'price': None, 
                'zones': 'Error fetching data', 
                'sector': crypto_info['category'],
                'company_name': crypto_info['name'],
                'industry': 'Cryptocurrency',
                'error': True,
                'asset_type': 'crypto'
            }
    
    def _calculate_crypto_period_resistance(self, closes, lookback_days):
        """Calculate resistance level for a given period using highest high and key resistance zones"""
        try:
            if len(closes) < lookback_days:
                lookback_days = len(closes)
            
            period_data = closes.tail(lookback_days)
            
            # Find the highest high in the period
            period_high = period_data.max()
            
            # Also consider areas where price has been rejected multiple times (resistance zones)
            # Calculate resistance as the 85th percentile of prices in the period (crypto is more volatile)
            resistance_zone = period_data.quantile(0.85)
            
            # Return the lower of the two (more conservative resistance)
            return min(period_high, resistance_zone)
            
        except Exception:
            return None

    def get_cryptos_near_support(self, threshold: float = 0.03) -> List[Dict]:
        """Get cryptocurrencies approaching support levels (from top crypto list only)"""
        results = []
        
        # Use top crypto list for analysis (no filtering since all crypto is just "Crypto" now)
        for symbol in self._top_crypto:
            try:
                result = self.check_crypto_support(symbol, threshold)
                if not result['error'] and result['zones'] not in (None, '', '—'):
                    results.append(result)
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                continue
        
        return results
    
    def get_cryptos_near_resistance(self, threshold: float = 0.03) -> List[Dict]:
        """Get cryptocurrencies approaching resistance levels (from top crypto list only)"""
        results = []
        
        # Use top crypto list for analysis
        for symbol in self._top_crypto:
            try:
                result = self.check_crypto_resistance(symbol, threshold)
                if not result['error'] and result['zones'] not in (None, '', '—'):
                    results.append(result)
            except Exception as e:
                logging.error(f"Error processing resistance for {symbol}: {e}")
                continue
        
        return results
    
    def get_detailed_crypto_analysis(self, symbol: str) -> Dict:
        """Get detailed technical analysis for a cryptocurrency"""
        try:
            crypto = yf.Ticker(symbol)
            hist = crypto.history(period="1y")
            
            if hist.empty:
                return {'error': 'No historical data available for this cryptocurrency'}
            
            closes = hist["Close"]
            volumes = hist["Volume"]
            current_price = closes.iloc[-1]
            
            # Calculate technical indicators
            ma21_series = closes.rolling(window=21).mean()
            ma50_series = closes.rolling(window=50).mean()
            ma200_series = closes.rolling(window=200).mean()
            
            ma21 = ma21_series.iloc[-1] if len(closes) >= 21 and not ma21_series.empty else None
            ma50 = ma50_series.iloc[-1] if len(closes) >= 50 and not ma50_series.empty else None
            ma200 = ma200_series.iloc[-1] if len(closes) >= 200 and not ma200_series.empty else None
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes)
            
            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(closes)
            
            # Calculate volatility (30-day)
            returns = closes.pct_change().dropna()
            volatility = returns.rolling(window=30).std().iloc[-1] * np.sqrt(365) * 100 if len(returns) >= 30 else None
            
            # Get crypto info
            crypto_info = self.get_crypto_info(symbol)
            
            # Check if crypto is near support for TradingView link
            support_check = self.check_crypto_support(symbol, 0.03)  # Use 3% threshold
            tradingview_link = support_check.get('tradingview_link') if not support_check.get('error') else None
            
            return {
                'symbol': symbol,
                'name': crypto_info['name'],
                'category': crypto_info['category'],
                'current_price': round(current_price, 4),
                'ma21': round(ma21, 4) if ma21 else None,
                'ma50': round(ma50, 4) if ma50 else None,
                'ma200': round(ma200, 4) if ma200 else None,
                'rsi': round(rsi, 2) if rsi else None,
                'macd': round(macd_line, 4) if macd_line else None,
                'macd_signal': round(signal_line, 4) if signal_line else None,
                'macd_histogram': round(histogram, 4) if histogram else None,
                'volatility': round(volatility, 2) if volatility else None,
                'volume': int(volumes.iloc[-1]) if not volumes.empty else None,
                'price_change_24h': round(((current_price - closes.iloc[-2]) / closes.iloc[-2] * 100), 2) if len(closes) >= 2 else None,
                'is_halal': True,  # All crypto assumed halal
                'tradingview_link': tradingview_link,
                'asset_type': 'crypto'
            }
        except Exception as e:
            logging.error(f"Error getting detailed analysis for {symbol}: {e}")
            return {'error': f'Unable to fetch detailed analysis for {symbol}'}
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        if len(prices) < window + 1:
            return None
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None
    
    def _calculate_macd(self, prices):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < 26:
            return None, None, None
        
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return (
            macd_line.iloc[-1] if not macd_line.empty else None,
            signal_line.iloc[-1] if not signal_line.empty else None,
            histogram.iloc[-1] if not histogram.empty else None
        )

# Global instance
crypto_manager = CryptoDataManager()