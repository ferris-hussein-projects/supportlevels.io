import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import threading
import time

class StockAnalyzer:
    def __init__(self):
        # Load full S&P 500 list
        self.ALL_SP500_TICKERS = self._load_sp500_tickers()

        # Default top 20 Fortune 500 companies by market cap and revenue
        self.DEFAULT_TOP_STOCKS = [
            'AAPL',  # Apple Inc.
            'MSFT',  # Microsoft Corporation
            'GOOGL', # Alphabet Inc. (Google)
            'AMZN',  # Amazon.com Inc.
            'TSLA',  # Tesla Inc.
            'META',  # Meta Platforms Inc. (Facebook)
            'NVDA',  # NVIDIA Corporation
            'BRK-B', # Berkshire Hathaway Inc.
            'JPM',   # JPMorgan Chase & Co.
            'JNJ',   # Johnson & Johnson
            'V',     # Visa Inc.
            'PG',    # Procter & Gamble Co.
            'HD',    # The Home Depot Inc.
            'MA',    # Mastercard Inc.
            'UNH',   # UnitedHealth Group Inc.
            'BAC',   # Bank of America Corp.
            'ADBE',  # Adobe Inc.
            'DIS',   # The Walt Disney Company
            'CRM',   # Salesforce Inc.
            'NFLX'   # Netflix Inc.
        ]

        # Current active top stocks (configurable)
        self._top_stocks = self.DEFAULT_TOP_STOCKS.copy()
        
        # Data cache system
        self._data_cache = {}
        self._cache_timestamp = None
        self._cache_lock = threading.Lock()
        self._cache_expiry_hours = 4  # Cache expires after 4 hours
        self._is_cache_warming = False
        
        logging.info(f"Loaded {len(self.ALL_SP500_TICKERS)} S&P 500 stocks, tracking top {len(self._top_stocks)}")
        
        # Start cache warming in background
        self._warm_cache_async()

    def _load_sp500_tickers(self) -> List[str]:
        """Load all S&P 500 tickers from Wikipedia"""
        try:
            # Fetch S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]

            # Clean and extract symbols
            tickers = df['Symbol'].str.replace('.', '-').tolist()  # Convert to Yahoo Finance format

            # Remove any invalid tickers
            tickers = [ticker for ticker in tickers if ticker and isinstance(ticker, str)]

            logging.info(f"Successfully loaded {len(tickers)} S&P 500 tickers")
            return sorted(tickers)

        except Exception as e:
            logging.error(f"Error loading S&P 500 tickers: {e}")
            # Fallback to a larger list of major stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ',
                'V', 'PG', 'HD', 'MA', 'UNH', 'BAC', 'ADBE', 'DIS', 'CRM', 'NFLX', 'XOM', 'CVX',
                'LLY', 'WMT', 'PFE', 'KO', 'ABBV', 'PEP', 'COST', 'TMO', 'VZ', 'T', 'CMCSA',
                'DHR', 'NEE', 'ACN', 'LIN', 'ABT', 'ORCL', 'WFC', 'MRK', 'AMD', 'CVS', 'BMY',
                'INTC', 'PM', 'HON', 'UPS', 'QCOM', 'LOW', 'RTX', 'IBM', 'TXN', 'GE', 'CAT',
                'SPGI', 'INTU', 'BA', 'MDT', 'GS', 'BLK', 'AXP', 'DE', 'SYK', 'C', 'MMM',
                'ISRG', 'BKNG', 'TJX', 'ADP', 'GILD', 'REGN', 'CB', 'MO', 'ZTS', 'SO', 'DUK',
                'CCI', 'TMUS', 'CSX', 'PLD', 'BDX', 'ITW', 'EOG', 'SHW', 'AON', 'EQIX', 'BSX',
                'FCX', 'APD', 'CL', 'EL', 'NSC', 'USB', 'MMC', 'PNC', 'ICE', 'CME', 'DG', 'F',
                'FDX', 'TGT', 'NOC', 'GM', 'GD', 'BIIB', 'EMR', 'OXY', 'PSA', 'SLB', 'ADI'
            ]

    @property
    def TICKERS(self) -> List[str]:
        """Get current top stocks being tracked"""
        return self._top_stocks

    def get_all_sp500_tickers(self) -> List[str]:
        """Get all S&P 500 tickers"""
        return self.ALL_SP500_TICKERS

    def get_top_stocks(self) -> List[str]:
        """Get current top stocks list"""
        return self._top_stocks.copy()

    def set_top_stocks(self, tickers: List[str]) -> bool:
        """Set new top stocks list (must be valid S&P 500 tickers)"""
        try:
            # Validate all tickers are in S&P 500
            invalid_tickers = [t for t in tickers if t not in self.ALL_SP500_TICKERS]
            if invalid_tickers:
                logging.warning(f"Invalid tickers not in S&P 500: {invalid_tickers}")
                return False

            self._top_stocks = tickers
            logging.info(f"Updated top stocks list to {len(tickers)} stocks")
            return True
        except Exception as e:
            logging.error(f"Error setting top stocks: {e}")
            return False

    def add_to_top_stocks(self, ticker: str) -> bool:
        """Add a stock to the top stocks list"""
        if ticker in self.ALL_SP500_TICKERS and ticker not in self._top_stocks:
            self._top_stocks.append(ticker)
            logging.info(f"Added {ticker} to top stocks")
            return True
        return False

    def remove_from_top_stocks(self, ticker: str) -> bool:
        """Remove a stock from the top stocks list"""
        if ticker in self._top_stocks:
            self._top_stocks.remove(ticker)
            logging.info(f"Removed {ticker} from top stocks")
            return True
        return False

    def reset_to_default_top_stocks(self):
        """Reset to default top 20 stocks"""
        self._top_stocks = self.DEFAULT_TOP_STOCKS.copy()
        logging.info("Reset to default top 20 stocks")
        # Warm cache with new stock list
        self._warm_cache_async()
    
    def _warm_cache_async(self):
        """Start cache warming in background thread"""
        if not self._is_cache_warming:
            thread = threading.Thread(target=self._warm_cache, daemon=True)
            thread.start()
    
    def _warm_cache(self):
        """Warm the cache with stock data at startup"""
        try:
            self._is_cache_warming = True
            logging.info("Starting cache warming...")
            
            # Warm cache for top stocks
            for ticker in self._top_stocks:
                try:
                    self._fetch_and_cache_stock_data(ticker)
                    time.sleep(0.1)  # Small delay to avoid hitting rate limits
                except Exception as e:
                    logging.error(f"Error caching data for {ticker}: {e}")
                    continue
            
            self._cache_timestamp = datetime.now()
            logging.info(f"Cache warming completed for {len(self._top_stocks)} stocks")
            
        except Exception as e:
            logging.error(f"Error during cache warming: {e}")
        finally:
            self._is_cache_warming = False
    
    def _fetch_and_cache_stock_data(self, ticker):
        """Fetch and cache stock data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y")
            
            if not hist.empty:
                with self._cache_lock:
                    self._data_cache[ticker] = {
                        'history': hist,
                        'timestamp': datetime.now()
                    }
                logging.debug(f"Cached data for {ticker}")
            else:
                logging.warning(f"No historical data for {ticker}")
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
            # Don't let one failed ticker break the entire caching process
            pass
    
    def _get_cached_data(self, ticker):
        """Get cached data for a ticker, or fetch if not available"""
        try:
            with self._cache_lock:
                # Check if we have cached data
                if ticker in self._data_cache:
                    cached_data = self._data_cache[ticker]
                    # Check if cache is still valid (within expiry time)
                    if self._cache_timestamp and (datetime.now() - self._cache_timestamp).seconds < (self._cache_expiry_hours * 3600):
                        hist = cached_data.get('history')
                        if hist is not None and not hist.empty:
                            return hist
                
                # Cache miss or expired - fetch new data
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5y")
                    
                    if not hist.empty:
                        self._data_cache[ticker] = {
                            'history': hist,
                            'timestamp': datetime.now()
                        }
                        return hist
                    else:
                        logging.warning(f"No historical data available for {ticker}")
                        return None
                except Exception as e:
                    logging.error(f"Error fetching data for {ticker}: {e}")
                    return None
        except Exception as e:
            logging.error(f"Critical error in _get_cached_data for {ticker}: {e}")
            return None
    
    def _is_cache_valid(self):
        """Check if cache is still valid"""
        if not self._cache_timestamp:
            return False
        return (datetime.now() - self._cache_timestamp).seconds < (self._cache_expiry_hours * 3600)

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        if len(prices) < window + 1:
            return None

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None

    def calculate_macd(self, prices):
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

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        if len(prices) < window:
            return None, None, None

        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        return (
            upper_band.iloc[-1] if not upper_band.empty else None,
            rolling_mean.iloc[-1] if not rolling_mean.empty else None,
            lower_band.iloc[-1] if not lower_band.empty else None
        )

    def check_support(self, ticker, threshold=0.001):
        """Check if stock is near historical support levels (1M, 6M, 1Y, 5Y)"""
        try:
            # Use cached data instead of fetching every time
            hist = self._get_cached_data(ticker)

            if hist is None or hist.empty or len(hist) < 30:
                return {'ticker': ticker, 'price': None, 'zones': 'Insufficient data', 'error': True}

            closes = hist["Close"]
            current_price = closes.iloc[-1]

            # Calculate support levels based on historical lows in different timeframes
            support_1m = self._calculate_period_support(closes, 21)  # ~1 month
            support_6m = self._calculate_period_support(closes, 126)  # ~6 months  
            support_1y = self._calculate_period_support(closes, 252)  # ~1 year
            support_5y = self._calculate_period_support(closes, len(closes))  # All available data

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
                        support_prices.append(round(support_price, 2))

            support_prices_str = ', '.join([f"${price}" for price in support_prices]) if support_prices else '—'

            # Generate TradingView link if asset is near support
            tradingview_link = None
            if zones:  # Only generate link if actually near support
                # Try to determine the exchange for TradingView
                tradingview_symbol = self._get_tradingview_symbol(ticker)
                tradingview_link = f"https://www.tradingview.com/chart/?symbol={tradingview_symbol}"

            return {
                'ticker': ticker,
                'price': round(current_price, 2),
                'zones': ', '.join(zones) or '—',
                'support_prices': support_prices_str,
                'support_levels': support_prices,  # Array for programmatic access
                'support_timeframes': zones,  # Array of timeframe names for pairing
                'support_1m': round(support_1m, 2) if support_1m else None,
                'support_6m': round(support_6m, 2) if support_6m else None,
                'support_1y': round(support_1y, 2) if support_1y else None,
                'support_5y': round(support_5y, 2) if support_5y else None,
                'volume': int(hist["Volume"].iloc[-1]) if not hist["Volume"].empty else None,
                'is_halal': self.is_stock_halal(ticker),
                'tradingview_link': tradingview_link,
                'level_type': 'support',
                'error': False
            }
        except Exception as e:
            logging.error(f"Error checking support for {ticker}: {e}")
            return {'ticker': ticker, 'price': None, 'zones': 'Error fetching data', 'error': True}

    def check_resistance(self, ticker, threshold=0.001):
        """Check if stock is near historical resistance levels (1M, 6M, 1Y, 5Y)"""
        try:
            # Use cached data instead of fetching every time
            hist = self._get_cached_data(ticker)

            if hist is None or hist.empty or len(hist) < 30:
                return {'ticker': ticker, 'price': None, 'zones': 'Insufficient data', 'error': True}

            closes = hist["Close"]
            current_price = closes.iloc[-1]

            # Calculate resistance levels based on historical highs in different timeframes
            resistance_1m = self._calculate_period_resistance(closes, 21)  # ~1 month
            resistance_6m = self._calculate_period_resistance(closes, 126)  # ~6 months  
            resistance_1y = self._calculate_period_resistance(closes, 252)  # ~1 year
            resistance_5y = self._calculate_period_resistance(closes, len(closes))  # All available data

            zones = []
            resistance_prices = []

            # Check if current price is near each resistance level
            # IMPORTANT: Price must be BELOW resistance level to be considered "near resistance"
            resistance_levels = [
                (resistance_1m, '1M Resistance'),
                (resistance_6m, '6M Resistance'), 
                (resistance_1y, '1Y Resistance'),
                (resistance_5y, '5Y Resistance')
            ]

            for resistance_price, period in resistance_levels:
                if resistance_price and current_price < resistance_price:  # Price must be below resistance
                    # Calculate how close price is to resistance (as percentage below resistance)
                    distance_from_resistance = (resistance_price - current_price) / resistance_price
                    if distance_from_resistance <= threshold:  # Within threshold distance below resistance
                        zones.append(period)
                        resistance_prices.append(round(resistance_price, 2))

            resistance_prices_str = ', '.join([f"${price}" for price in resistance_prices]) if resistance_prices else '—'

            # Generate TradingView link if asset is near resistance
            tradingview_link = None
            if zones:  # Only generate link if actually near resistance
                # Try to determine the exchange for TradingView
                tradingview_symbol = self._get_tradingview_symbol(ticker)
                tradingview_link = f"https://www.tradingview.com/chart/?symbol={tradingview_symbol}"

            return {
                'ticker': ticker,
                'price': round(current_price, 2),
                'zones': ', '.join(zones) or '—',
                'resistance_prices': resistance_prices_str,
                'resistance_levels': resistance_prices,  # Array for programmatic access
                'resistance_1m': round(resistance_1m, 2) if resistance_1m else None,
                'resistance_6m': round(resistance_6m, 2) if resistance_6m else None,
                'resistance_1y': round(resistance_1y, 2) if resistance_1y else None,
                'resistance_5y': round(resistance_5y, 2) if resistance_5y else None,
                'volume': int(hist["Volume"].iloc[-1]) if not hist["Volume"].empty else None,
                'is_halal': self.is_stock_halal(ticker),
                'tradingview_link': tradingview_link,
                'level_type': 'resistance',
                'error': False
            }
        except Exception as e:
            logging.error(f"Error checking resistance for {ticker}: {e}")
            return {'ticker': ticker, 'price': None, 'zones': 'Error fetching data', 'error': True}

    def _get_tradingview_symbol(self, ticker):
        """Get the appropriate TradingView symbol format for a stock ticker"""

        # Handle special cases and exchange mappings
        ticker_clean = ticker.replace('-', '.')  # Convert BRK-B to BRK.B

        # Major exchange mappings for popular tickers
        nasdaq_tickers = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'ADBE', 'NFLX',
            'INTC', 'AMD', 'QCOM', 'COST', 'CMCSA', 'PEP', 'TMUS', 'INTU', 'CRM', 'PYPL'
        }

        nyse_tickers = {
            'JPM', 'JNJ', 'V', 'PG', 'HD', 'MA', 'UNH', 'BAC', 'DIS', 'WMT', 'VZ', 'KO',
            'XOM', 'CVX', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'BRK.B',
            'WFC', 'T', 'PM', 'NEE', 'HON', 'UPS', 'CAT', 'GE', 'F', 'GM'
        }

        if ticker in nasdaq_tickers:
            return f"NASDAQ:{ticker}"
        elif ticker_clean in nyse_tickers:
            return f"NYSE:{ticker_clean}"
        else:
            # Default to NASDAQ for unknown tickers
            return f"NASDAQ:{ticker}"

    def _calculate_period_support(self, closes, lookback_days):
        """Calculate support level using pivot lows, volume analysis, and price clustering"""
        try:
            if len(closes) < lookback_days:
                lookback_days = len(closes)

            period_data = closes.tail(lookback_days)
            if len(period_data) < 10:  # Need minimum data points
                return None

            # Get volume data if available
            hist = self._get_cached_data(closes.name if hasattr(closes, 'name') else 'UNKNOWN')
            volumes = None
            if hist is not None and 'Volume' in hist.columns:
                volumes = hist['Volume'].tail(lookback_days)

            # Method 1: Find pivot lows (swing lows)
            pivot_lows = self._find_pivot_lows(period_data, window=5)
            
            # Method 2: Price clustering - find areas where price spent significant time
            price_clusters = self._find_price_clusters(period_data, volumes)
            
            # Method 3: Traditional support levels
            period_low = period_data.min()
            support_percentiles = [period_data.quantile(q) for q in [0.05, 0.10, 0.15]]
            
            # Combine all potential support levels
            all_supports = []
            
            # Add pivot lows with higher weight
            all_supports.extend([(price, 3.0) for price in pivot_lows])
            
            # Add price clusters with medium weight
            all_supports.extend([(price, 2.0) for price in price_clusters])
            
            # Add traditional levels with lower weight
            all_supports.extend([(price, 1.0) for price in support_percentiles])
            all_supports.append((period_low, 1.5))
            
            if not all_supports:
                return period_low
            
            # Find the most significant support level
            # Group nearby levels and weight by significance
            current_price = closes.iloc[-1]
            significant_supports = []
            
            for price, weight in all_supports:
                if price and price < current_price:  # Only consider levels below current price
                    # Check if this level has been tested multiple times
                    test_count = self._count_support_tests(period_data, price, tolerance=0.02)
                    final_weight = weight * (1 + test_count * 0.5)
                    significant_supports.append((price, final_weight))
            
            if not significant_supports:
                return period_low
            
            # Return the support level with highest combined weight
            # But prefer levels that are not too far from current price
            best_support = None
            best_score = 0
            
            for price, weight in significant_supports:
                # Distance penalty - prefer levels closer to current price
                distance_factor = max(0.1, 1 - abs(current_price - price) / current_price)
                score = weight * distance_factor
                
                if score > best_score:
                    best_score = score
                    best_support = price
            
            return best_support if best_support else period_low

        except Exception as e:
            logging.error(f"Error in _calculate_period_support: {e}")
            return None

    def _find_pivot_lows(self, prices, window=5):
        """Find pivot lows (swing lows) in price data"""
        pivot_lows = []
        try:
            for i in range(window, len(prices) - window):
                current_price = prices.iloc[i]
                is_pivot_low = True
                
                # Check if current price is lower than surrounding prices
                for j in range(i - window, i + window + 1):
                    if j != i and prices.iloc[j] <= current_price:
                        is_pivot_low = False
                        break
                
                if is_pivot_low:
                    pivot_lows.append(current_price)
            
            return pivot_lows
        except Exception:
            return []
    
    def _find_price_clusters(self, prices, volumes=None, num_clusters=3):
        """Find price levels where price spent significant time (clustering)"""
        try:
            # Create price buckets
            price_range = prices.max() - prices.min()
            if price_range == 0:
                return []
            
            bucket_size = price_range / 50  # 50 buckets
            price_counts = {}
            
            for i, price in enumerate(prices):
                bucket = int((price - prices.min()) / bucket_size)
                bucket_price = prices.min() + bucket * bucket_size
                
                # Weight by volume if available
                weight = volumes.iloc[i] if volumes is not None else 1
                price_counts[bucket_price] = price_counts.get(bucket_price, 0) + weight
            
            # Find top clusters
            sorted_clusters = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)
            return [price for price, count in sorted_clusters[:num_clusters]]
            
        except Exception:
            return []
    
    def _count_support_tests(self, prices, support_level, tolerance=0.02):
        """Count how many times price has tested a support level"""
        try:
            test_count = 0
            support_range = support_level * tolerance
            
            for price in prices:
                if abs(price - support_level) <= support_range:
                    test_count += 1
            
            return min(test_count, 10)  # Cap at 10 for scoring purposes
        except Exception:
            return 0

    def _find_pivot_highs(self, prices, window=5):
        """Find pivot highs (swing highs) in price data"""
        pivot_highs = []
        try:
            for i in range(window, len(prices) - window):
                current_price = prices.iloc[i]
                is_pivot_high = True
                
                # Check if current price is higher than surrounding prices
                for j in range(i - window, i + window + 1):
                    if j != i and prices.iloc[j] >= current_price:
                        is_pivot_high = False
                        break
                
                if is_pivot_high:
                    pivot_highs.append(current_price)
            
            return pivot_highs
        except Exception:
            return []
    
    def _find_resistance_price_clusters(self, prices, volumes=None, num_clusters=3):
        """Find price levels where price spent significant time at resistance levels (clustering)"""
        try:
            # Focus on upper price ranges for resistance
            price_75th = prices.quantile(0.75)
            high_prices = prices[prices >= price_75th]
            
            if len(high_prices) < 5:
                return []
            
            # Create price buckets for high price range
            price_range = high_prices.max() - high_prices.min()
            if price_range == 0:
                return []
            
            bucket_size = price_range / 30  # 30 buckets
            price_counts = {}
            
            for i, price in enumerate(high_prices):
                bucket = int((price - high_prices.min()) / bucket_size)
                bucket_price = high_prices.min() + bucket * bucket_size
                
                # Weight by volume if available
                if volumes is not None and i < len(volumes):
                    weight = volumes.iloc[high_prices.index[i]] if high_prices.index[i] < len(volumes) else 1
                else:
                    weight = 1
                price_counts[bucket_price] = price_counts.get(bucket_price, 0) + weight
            
            # Find top clusters
            sorted_clusters = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)
            return [price for price, count in sorted_clusters[:num_clusters]]
            
        except Exception:
            return []
    
    def _count_resistance_tests(self, prices, resistance_level, tolerance=0.02):
        """Count how many times price has tested a resistance level"""
        try:
            test_count = 0
            resistance_range = resistance_level * tolerance
            
            for price in prices:
                if abs(price - resistance_level) <= resistance_range:
                    test_count += 1
            
            return min(test_count, 10)  # Cap at 10 for scoring purposes
        except Exception:
            return 0

    def _calculate_period_resistance(self, closes, lookback_days):
        """Calculate resistance level using pivot highs, volume analysis, and price clustering (enhanced method)"""
        try:
            if len(closes) < lookback_days:
                lookback_days = len(closes)

            period_data = closes.tail(lookback_days)
            if len(period_data) < 10:  # Need minimum data points
                return None

            # Get volume data if available
            hist = self._get_cached_data(closes.name if hasattr(closes, 'name') else 'UNKNOWN')
            volumes = None
            if hist is not None and 'Volume' in hist.columns:
                volumes = hist['Volume'].tail(lookback_days)

            # Method 1: Find pivot highs (swing highs)
            pivot_highs = self._find_pivot_highs(period_data, window=5)
            
            # Method 2: Price clustering - find areas where price spent significant time at high levels
            price_clusters = self._find_resistance_price_clusters(period_data, volumes)
            
            # Method 3: Traditional resistance levels
            period_high = period_data.max()
            resistance_percentiles = [period_data.quantile(q) for q in [0.95, 0.90, 0.85]]
            
            # Combine all potential resistance levels
            all_resistances = []
            
            # Add pivot highs with higher weight
            all_resistances.extend([(price, 3.0) for price in pivot_highs])
            
            # Add price clusters with medium weight
            all_resistances.extend([(price, 2.0) for price in price_clusters])
            
            # Add traditional levels with lower weight
            all_resistances.extend([(price, 1.0) for price in resistance_percentiles])
            all_resistances.append((period_high, 1.5))
            
            if not all_resistances:
                return period_high
            
            # Find the most significant resistance level
            current_price = closes.iloc[-1]
            significant_resistances = []
            
            for price, weight in all_resistances:
                if price and price > current_price:  # Only consider levels above current price
                    # Check if this level has been tested multiple times
                    test_count = self._count_resistance_tests(period_data, price, tolerance=0.02)
                    final_weight = weight * (1 + test_count * 0.5)
                    significant_resistances.append((price, final_weight))
            
            if not significant_resistances:
                return period_high
            
            # Return the resistance level with highest combined weight
            # But prefer levels that are not too far from current price
            best_resistance = None
            best_score = 0
            
            for price, weight in significant_resistances:
                # Distance penalty - prefer levels closer to current price
                distance_factor = max(0.1, 1 - abs(price - current_price) / current_price)
                score = weight * distance_factor
                
                if score > best_score:
                    best_score = score
                    best_resistance = price
            
            return best_resistance if best_resistance else period_high

        except Exception as e:
            logging.error(f"Error in _calculate_period_resistance: {e}")
            return None

    def get_stocks_near_levels(self, support_threshold=0.001, resistance_threshold=0.001, level_type='support', sector_filter='All', include_crypto=True):
        """Get stocks and optionally crypto approaching support or resistance levels with filtering and favorites support"""
        results = []
        
        logging.info(f"Starting get_stocks_near_levels with {len(self._top_stocks)} top stocks")

        try:
            # Handle favorites filter
            if sector_filter == 'Favorites':
                try:
                    from models import UserFavorites
                    favorite_tickers = UserFavorites.get_favorite_tickers()
                    if not favorite_tickers:
                        return results

                    # Process favorite stocks (limit to 50 for performance)
                    favorite_tickers = favorite_tickers[:50]
                    for ticker in favorite_tickers:
                        try:
                            if ticker in self.ALL_SP500_TICKERS:  # Check against full S&P 500 list
                                if level_type == 'support':
                                    result = self.check_support(ticker, support_threshold)
                                else:
                                    result = self.check_resistance(ticker, resistance_threshold)
                                
                                if not result.get('error') and result.get('zones') not in (None, '', '—'):
                                    result['asset_type'] = 'stock'
                                    result['sector'] = self.get_stock_sector(ticker)
                                    result['company_name'] = self.get_company_name(ticker)
                                    results.append(result)
                            else:
                                # Check if it's a crypto
                                from crypto_data import crypto_manager
                                crypto_symbols = crypto_manager.get_all_crypto_symbols()
                                if ticker in crypto_symbols:
                                    if level_type == 'support':
                                        crypto_result = crypto_manager.check_crypto_support(ticker, support_threshold)
                                    else:
                                        crypto_result = crypto_manager.check_crypto_resistance(ticker, resistance_threshold)
                                    if crypto_result and not crypto_result.get('error'):
                                        results.append(crypto_result)
                        except Exception as ticker_error:
                            logging.error(f"Error processing favorite ticker {ticker}: {ticker_error}")
                            continue
                    return results
                except Exception as fav_error:
                    logging.error(f"Error processing favorites: {fav_error}")
                    return results

            # Get stock results from top stocks only
            tickers_to_check = self._top_stocks[:30]  # Limit to 30 for performance
            if sector_filter != 'All' and not sector_filter.startswith('Crypto'):
                # Filter by stock sector
                tickers_to_check = [ticker for ticker in self._top_stocks if self.get_stock_sector(ticker) == sector_filter][:20]
            elif sector_filter.startswith('Crypto'):
                # Only crypto filter selected, skip stocks
                tickers_to_check = []

            # Process stocks in batches for better performance
            batch_size = 10
            for i in range(0, len(tickers_to_check), batch_size):
                batch = tickers_to_check[i:i + batch_size]
                for ticker in batch:
                    try:
                        if level_type == 'support':
                            result = self.check_support(ticker, support_threshold)
                        else:
                            result = self.check_resistance(ticker, resistance_threshold)
                            
                        if not result.get('error') and result.get('zones') not in (None, '', '—'):
                            result['sector'] = self.get_stock_sector(ticker)
                            result['company_name'] = self.get_company_name(ticker)
                            result['asset_type'] = 'stock'
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Error processing {ticker}: {e}")
                        continue

            # Get crypto results if enabled
            if include_crypto:
                try:
                    from crypto_data import crypto_manager
                    if sector_filter == 'Crypto':
                        # Only crypto selected
                        if level_type == 'support':
                            crypto_results = crypto_manager.get_cryptos_near_support(support_threshold)
                        else:
                            crypto_results = crypto_manager.get_cryptos_near_resistance(resistance_threshold)
                        results.extend(crypto_results or [])
                    elif sector_filter == 'All':
                        # All assets selected, include crypto
                        if level_type == 'support':
                            crypto_results = crypto_manager.get_cryptos_near_support(support_threshold)
                        else:
                            crypto_results = crypto_manager.get_cryptos_near_resistance(resistance_threshold)
                        results.extend(crypto_results or [])
                    # If a specific stock sector is selected, skip crypto
                except Exception as crypto_error:
                    logging.error(f"Error processing crypto data: {crypto_error}")

        except Exception as e:
            logging.error(f"Critical error in get_stocks_near_levels: {e}")
            return []

        return results

    def get_stocks_near_support(self, threshold=0.001, sector_filter='All', include_crypto=True):
        """Backward compatibility method - use get_stocks_near_levels instead"""
        return self.get_stocks_near_levels(support_threshold=threshold, level_type='support', sector_filter=sector_filter, include_crypto=include_crypto)

    def get_stock_sector(self, ticker):
        """Get sector for any S&P 500 stock (enhanced from get_fortune500_sector)"""
        # Conservative sector mapping for major stocks
        conservative_sectors = {
            # Technology Giants
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'META': 'Technology',
            'NVDA': 'Technology',
            'ADBE': 'Technology',
            'CRM': 'Technology',
            'NFLX': 'Technology',  # More conservative - streaming is tech
            'ORCL': 'Technology',
            'INTC': 'Technology',
            'AMD': 'Technology',
            'QCOM': 'Technology',
            'IBM': 'Technology',
            'TXN': 'Technology',
            'ACN': 'Technology',
            'INTU': 'Technology',

            # Consumer & Retail
            'AMZN': 'Consumer',
            'TSLA': 'Consumer',
            'HD': 'Consumer',
            'WMT': 'Consumer',
            'COST': 'Consumer',
            'TJX': 'Consumer',
            'LOW': 'Consumer',
            'TGT': 'Consumer',
            'DIS': 'Consumer',  # Entertainment = consumer
            'PG': 'Consumer',
            'KO': 'Consumer',
            'PEP': 'Consumer',
            'PM': 'Consumer',
            'MO': 'Consumer',
            'CL': 'Consumer',
            'EL': 'Consumer',

            # Financial Services
            'BRK-B': 'Financial',
            'JPM': 'Financial',
            'V': 'Financial',
            'MA': 'Financial',
            'BAC': 'Financial',
            'WFC': 'Financial',
            'GS': 'Financial',
            'BLK': 'Financial',
            'AXP': 'Financial',
            'USB': 'Financial',
            'PNC': 'Financial',
            'C': 'Financial',
            'ICE': 'Financial',
            'CME': 'Financial',
            'MMC': 'Financial',
            'AON': 'Financial',
            'CB': 'Financial',
            'SPGI': 'Financial',

            # Healthcare
            'JNJ': 'Healthcare',
            'UNH': 'Healthcare',
            'PFE': 'Healthcare',
            'ABBV': 'Healthcare',
            'LLY': 'Healthcare',
            'MRK': 'Healthcare',
            'TMO': 'Healthcare',
            'ABT': 'Healthcare',
            'DHR': 'Healthcare',
            'BMY': 'Healthcare',
            'GILD': 'Healthcare',
            'REGN': 'Healthcare',
            'BIIB': 'Healthcare',
            'ZTS': 'Healthcare',
            'BDX': 'Healthcare',
            'BSX': 'Healthcare',
            'ISRG': 'Healthcare',
            'MDT': 'Healthcare',
            'SYK': 'Healthcare',

            # Energy & Materials
            'XOM': 'Energy',
            'CVX': 'Energy',
            'EOG': 'Energy',
            'SLB': 'Energy',
            'OXY': 'Energy',
            'FCX': 'Energy',  # Mining = energy sector
            'LIN': 'Energy',  # Industrial gases = energy
            'APD': 'Energy',

            # Industrial
            'HON': 'Industrial',
            'UPS': 'Industrial',
            'RTX': 'Industrial',
            'GE': 'Industrial',
            'CAT': 'Industrial',
            'ITW': 'Industrial',
            'MMM': 'Industrial',
            'BA': 'Industrial',
            'CSX': 'Industrial',
            'NSC': 'Industrial',
            'FDX': 'Industrial',
            'NOC': 'Industrial',
            'GD': 'Industrial',
            'EMR': 'Industrial',
            'DE': 'Industrial',
            'LMT': 'Industrial',

            # Communication & Media
            'VZ': 'Communication',
            'T': 'Communication',
            'CMCSA': 'Communication',
            'TMUS': 'Communication',

            # Utilities & REITs
            'NEE': 'Utilities',
            'SO': 'Utilities',
            'DUK': 'Utilities',
            'PLD': 'Utilities',  # REITs grouped with utilities
            'CCI': 'Utilities',  # Cell towers = utilities
            'EQIX': 'Utilities',  # Data centers = utilities
            'PSA': 'Utilities',  # Storage REITs = utilities

            # Automotive
            'F': 'Industrial',  # Auto = industrial
            'GM': 'Industrial',

            # Hospitality & Travel
            'BKNG': 'Consumer',  # Travel = consumer

            # Other Services
            'ADP': 'Technology',  # Payroll tech = technology
            'DG': 'Consumer',  # Dollar stores = consumer
            'CVS': 'Healthcare',  # Pharmacy = healthcare
            'SHW': 'Industrial'  # Paint = industrial
        }

        if ticker in conservative_sectors:
            return conservative_sectors[ticker]

        # For other S&P 500 stocks, try to get from yfinance but map to conservative categories
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', 'Other')

            # Map detailed sectors to conservative ones
            sector_mapping = {
                'Information Technology': 'Technology',
                'Technology': 'Technology',
                'Health Care': 'Healthcare',
                'Healthcare': 'Healthcare',
                'Financials': 'Financial',
                'Financial Services': 'Financial',
                'Consumer Discretionary': 'Consumer',
                'Consumer Staples': 'Consumer',
                'Consumer Cyclical': 'Consumer',
                'Consumer Defensive': 'Consumer',
                'Communication Services': 'Communication',
                'Industrials': 'Industrial',
                'Energy': 'Energy',
                'Materials': 'Energy',  # Materials grouped with energy
                'Real Estate': 'Utilities',  # REITs grouped with utilities
                'Utilities': 'Utilities'
            }

            return sector_mapping.get(sector, 'Other')

        except Exception:
            return 'Other'

    # Keep the old method for backward compatibility
    def get_fortune500_sector(self, ticker):
        """Backward compatibility - use get_stock_sector instead"""
        return self.get_stock_sector(ticker)

    def get_company_name(self, ticker):
        """Get company name for Fortune 500 top 20 stocks"""
        company_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc. (Google)',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'BRK-B': 'Berkshire Hathaway Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.',
            'PG': 'Procter & Gamble Co.',
            'HD': 'The Home Depot Inc.',
            'MA': 'Mastercard Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'BAC': 'Bank of America Corp.',
            'ADBE': 'Adobe Inc.',
            'DIS': 'The Walt Disney Company',
            'CRM': 'Salesforce Inc.',
            'NFLX': 'Netflix Inc.'
        }
        return company_names.get(ticker, ticker)

    def get_detailed_analysis(self, ticker):
        """Get detailed technical analysis for a stock"""
        try:
            # Use cached data for history, but still fetch info directly for latest details
            hist = self._get_cached_data(ticker)
            if hist is not None and len(hist) >= 252:  # If we have enough cached data, use last year
                hist = hist.tail(252)  # Use last 252 days (1 year)
            
            stock = yf.Ticker(ticker)
            info = stock.info

            if hist is None or hist.empty:
                return {'error': 'No historical data available for this ticker'}

            closes = hist["Close"]
            volumes = hist["Volume"]
            current_price = closes.iloc[-1]

            # Moving averages
            ma21 = closes.rolling(window=21).mean().iloc[-1] if len(closes) >= 21 else None
            ma50 = closes.rolling(window=50).mean().iloc[-1] if len(closes) >= 50 else None
            ma200 = closes.rolling(window=200).mean().iloc[-1] if len(closes) >= 200 else None

            # Technical indicators
            rsi = self.calculate_rsi(closes)
            macd, signal, histogram = self.calculate_macd(closes)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)

            # Performance metrics
            price_1w = closes.iloc[-5] if len(closes) >= 5 else None
            price_1m = closes.iloc[-21] if len(closes) >= 21 else None
            price_3m = closes.iloc[-63] if len(closes) >= 63 else None

            change_1w = ((current_price - price_1w) / price_1w * 100) if price_1w else None
            change_1m = ((current_price - price_1m) / price_1m * 100) if price_1m else None
            change_3m = ((current_price - price_3m) / price_3m * 100) if price_3m else None

            # Volume analysis
            avg_volume_30d = volumes.tail(30).mean() if len(volumes) >= 30 else None
            current_volume = volumes.iloc[-1] if not volumes.empty else None

            # Check if asset is near support for TradingView link
            support_check = self.check_support(ticker, 0.03)  # Use 3% threshold
            tradingview_link = support_check.get('tradingview_link') if not support_check.get('error') else None

            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'current_price': round(current_price, 2),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap'),
                'volume': int(current_volume) if current_volume else None,
                'avg_volume_30d': int(avg_volume_30d) if avg_volume_30d else None,

                # Moving averages
                'ma21': round(ma21, 2) if ma21 else None,
                'ma50': round(ma50, 2) if ma50 else None,
                'ma200': round(ma200, 2) if ma200 else None,

                # Technical indicators
                'rsi': round(rsi, 2) if rsi else None,
                'macd': round(macd, 4) if macd else None,
                'macd_signal': round(signal, 4) if signal else None,
                'macd_histogram': round(histogram, 4) if histogram else None,

                # Bollinger Bands
                'bb_upper': round(bb_upper, 2) if bb_upper else None,
                'bb_middle': round(bb_middle, 2) if bb_middle else None,
                'bb_lower': round(bb_lower, 2) if bb_lower else None,

                # Performance
                'change_1w': round(change_1w, 2) if change_1w else None,
                'change_1m': round(change_1m, 2) if change_1m else None,
                'change_3m': round(change_3m, 2) if change_3m else None,

                # Additional info
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'is_halal': self.is_stock_halal(ticker),
                'tradingview_link': tradingview_link,

                'error': None
            }
        except Exception as e:
            logging.error(f"Error getting detailed analysis for {ticker}: {e}")
            return {'error': f'Unable to fetch data for {ticker}. Please verify the ticker symbol.'}

    def get_chart_data(self, ticker, period='6mo'):
        """Get chart data for a stock"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                return {'error': 'No data available'}

            # Prepare data for Chart.js
            dates = [date.strftime('%Y-%m-%d') for date in hist.index]
            prices = hist['Close'].round(2).tolist()
            volumes = hist['Volume'].tolist()

            # Calculate moving averages for chart
            ma21 = hist['Close'].rolling(window=21).mean().round(2).tolist()
            ma50 = hist['Close'].rolling(window=50).mean().round(2).tolist()

            return {
                'dates': dates,
                'prices': prices,
                'volumes': volumes,
                'ma21': ma21,
                'ma50': ma50,
                'ticker': ticker
            }
        except Exception as e:
            logging.error(f"Error getting chart data for {ticker}: {e}")
            return {'error': 'Unable to fetch chart data'}

    def get_dashboard_data(self, threshold=0.03):
        """Get dashboard summary data"""
        try:
            support_stocks = self.get_stocks_near_support(threshold)

            # Market overview - get data for major indices
            indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
            market_data = {}

            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period='5d')
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2] if len(hist) >= 2 else current
                        change = ((current - previous) / previous * 100) if previous != 0 else 0

                        market_data[index] = {
                            'value': round(current, 2),
                            'change': round(change, 2)
                        }
                except Exception:
                    continue

            # Sector analysis
            sector_counts = {}
            for stock in support_stocks[:20]:  # Limit for performance
                try:
                    analysis = self.get_detailed_analysis(stock['ticker'])
                    if not analysis.get('error') and analysis.get('sector'):
                        sector = analysis['sector']
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                except Exception:
                    continue

            return {
                'total_stocks_available': len(self.ALL_SP500_TICKERS),
                'total_stocks_tracked': len(self._top_stocks),
                'stocks_near_support': len(support_stocks),
                'support_percentage': round(len(support_stocks) / len(self._top_stocks) * 100, 1) if self._top_stocks else 0,
                'market_indices': market_data,
                'sector_breakdown': sector_counts,
                'top_support_stocks': support_stocks[:10]
            }
        except Exception as e:
            logging.error(f"Error getting dashboard data: {e}")
            return {'error': 'Unable to fetch dashboard data'}

    def compare_stocks(self, tickers):
        """Compare multiple stocks"""
        try:
            comparison_data = {}

            for ticker in tickers:
                analysis = self.get_detailed_analysis(ticker)
                if not analysis.get('error'):
                    comparison_data[ticker] = analysis

            if not comparison_data:
                return {'error': 'No valid data found for the selected stocks'}

            return {
                'stocks': comparison_data,
                'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logging.error(f"Error comparing stocks {tickers}: {e}")
            return {'error': 'Unable to compare selected stocks'}

    def is_stock_halal(self, ticker):
        """Check if a stock is halal-compliant based on Islamic finance principles"""

        # Non-halal companies based on business activities
        non_halal_stocks = {
            # Alcohol & Beverages
            'BUD', 'TAP', 'STZ', 'DEO', 'COKE',

            # Tobacco
            'PM', 'MO', 'BTI',

            # Gambling & Casinos
            'LVS', 'WYNN', 'MGM', 'CZR', 'PENN', 'DKNG',

            # Adult Entertainment / Questionable Content
            # (Most major streaming/media considered acceptable if diversified)

            # Conventional Banking (pure interest-based business)
            'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'COF',
            'MS', 'GS', 'AXP',  # Investment banking with significant interest income

            # Insurance (conventional insurance with interest/gambling elements)
            'BRK-B',  # Berkshire has insurance and other non-halal investments
            'AIG', 'PRU', 'MET', 'AFL', 'ALL', 'TRV', 'PGR',

            # Defense/Weapons (primary business)
            'LMT', 'RTX', 'NOC', 'GD', 'BA',  # Boeing has significant defense

            # Pork-related
            'TSN',  # Tyson Foods (major pork producer)
            'HRL',  # Hormel (pork products)

            # Adult/Questionable Entertainment
            'NWSA', 'FOXA'  # News Corp/Fox (some scholars avoid due to content)
        }

        if ticker in non_halal_stocks:
            return False

        # Companies generally considered halal-compliant
        halal_stocks = {
            # Technology (generally halal)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL',
            'INTC', 'AMD', 'QCOM', 'IBM', 'TXN', 'ACN', 'INTU', 'NOW', 'PANW',

            # Healthcare (generally halal)
            'JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'ABT', 'DHR',
            'BMY', 'GILD', 'REGN', 'BIIB', 'ZTS', 'BDX', 'BSX', 'ISRG', 'MDT', 'SYK',
            'CVS',  # Pharmacy services

            # Consumer Goods (halal products)
            'PG', 'KO', 'PEP', 'CL', 'EL', 'NKE', 'COST', 'WMT', 'TGT', 'HD', 'LOW',
            'AMZN',  # E-commerce and cloud (diversified, no primary haram business)
            'TSLA',  # Electric vehicles
            'TJX', 'DG', 'SBUX',

            # Industrials (generally halal manufacturing)
            'HON', 'UPS', 'FDX', 'CAT', 'DE', 'EMR', 'ITW', 'MMM', 'GE',
            'CSX', 'NSC',  # Transportation
            'PCAR', 'F', 'GM',  # Automotive

            # Energy (oil/gas - generally considered halal)
            'XOM', 'CVX', 'EOG', 'SLB', 'OXY', 'COP', 'PSX', 'VLO', 'MPC',

            # Materials & Chemicals
            'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'SHW',

            # Utilities (generally halal)
            'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'SRE', 'D', 'PCG',

            # Communication/Telecom
            'VZ', 'T', 'TMUS', 'CMCSA',  # Infrastructure services

            # Media & Entertainment (diversified, not primarily haram)
            'DIS',  # Disney (theme parks, movies - generally accepted)
            'NFLX',  # Netflix (streaming service - content is diverse)

            # REITs (real estate generally halal if not hotels/casinos)
            'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR',

            # Financial Services (Islamic compliant or minimal interest exposure)
            'V', 'MA',  # Payment processors (fee-based, not interest-based)
            'PYPL', 'SQ',  # Payment technology
            'SPGI', 'MCO',  # Credit rating (information services)
            'ICE', 'CME',  # Exchanges (fee-based)
            'BLK',  # Asset management (though some avoid due to interest investments)

            # Travel & Hospitality (generally halal)
            'BKNG',  # Booking services
            'AAL', 'DAL', 'UAL', 'LUV',  # Airlines

            # Food & Beverages (halal food companies)
            'MCD', 'YUM', 'QSR', 'MDLZ', 'GIS', 'K', 'CPB', 'SJM',
            # Note: McDonald's, KFC etc. have halal options in Muslim countries
        }

        if ticker in halal_stocks:
            return True

        # For unknown stocks, try to determine based on sector and business
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            sector = info.get('sector', '')
            industry = info.get('industry', '')
            business_summary = info.get('longBusinessSummary', '').lower()

            # Check for non-halal keywords in business description
            non_halal_keywords = [
                'alcohol', 'beer', 'wine', 'liquor', 'tobacco', 'cigarette',
                'casino', 'gambling', 'gaming', 'lottery', 'adult entertainment',
                'pork', 'bacon', 'ham', 'insurance', 'bank', 'financial services',
                'defense', 'weapons', 'military', 'ammunition'
            ]

            for keyword in non_halal_keywords:
                if keyword in business_summary:
                    return False

            # Generally halal sectors
            halal_sectors = [
                'Technology', 'Healthcare', 'Consumer Defensive', 'Consumer Cyclical',
                'Industrials', 'Energy', 'Basic Materials', 'Utilities',
                'Communication Services', 'Real Estate'
            ]

            if sector in halal_sectors:
                return True

            # Default to uncertain (marked as non-halal for conservative approach)
            return False

        except Exception:
            # If we can't determine, be conservative
            return False