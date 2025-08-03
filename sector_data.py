import requests
import pandas as pd
import yfinance as yf
import json
from typing import Dict, List, Optional

class SectorDataManager:
    """Manages S&P 500 sector and industry classification data"""
    
    def __init__(self):
        self.sp500_data = None
        self.sector_mapping = {}
        
    def fetch_sp500_with_sectors(self) -> pd.DataFrame:
        """Fetch complete S&P 500 list with sector information"""
        try:
            # Primary source: Wikipedia S&P 500 list with sectors
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            
            # Get the main table with current S&P 500 companies
            df = tables[0]
            
            # Clean and standardize column names
            df.columns = ['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry', 'Headquarters', 'Date_Added', 'CIK', 'Founded']
            
            # Clean the data
            df = df.dropna(subset=['Symbol', 'GICS_Sector'])
            df['Symbol'] = df['Symbol'].str.strip()
            df['GICS_Sector'] = df['GICS_Sector'].str.strip()
            df['GICS_Sub_Industry'] = df['GICS_Sub_Industry'].str.strip()
            
            # Create simplified sector categories
            sector_simplification = {
                'Information Technology': 'Technology',
                'Health Care': 'Healthcare', 
                'Financials': 'Financial',
                'Communication Services': 'Communication',
                'Consumer Discretionary': 'Consumer Discretionary',
                'Consumer Staples': 'Consumer Staples',
                'Industrials': 'Industrial',
                'Energy': 'Energy',
                'Materials': 'Materials',
                'Real Estate': 'Real Estate',
                'Utilities': 'Utilities'
            }
            
            df['Sector'] = df['GICS_Sector'].map(sector_simplification).fillna(df['GICS_Sector'])
            
            self.sp500_data = df
            self._create_sector_mapping()
            
            return df
            
        except Exception as e:
            print(f"Error fetching S&P 500 data: {e}")
            # Fallback to basic list
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> pd.DataFrame:
        """Fallback S&P 500 data if primary source fails"""
        # Basic S&P 500 tickers with estimated sectors
        fallback_data = [
            ('AAPL', 'Apple Inc.', 'Technology'),
            ('MSFT', 'Microsoft Corporation', 'Technology'),
            ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary'),
            ('GOOGL', 'Alphabet Inc. Class A', 'Technology'),
            ('GOOG', 'Alphabet Inc. Class C', 'Technology'),
            ('META', 'Meta Platforms Inc.', 'Technology'),
            ('TSLA', 'Tesla Inc.', 'Consumer Discretionary'),
            ('BRK-B', 'Berkshire Hathaway Inc. Class B', 'Financial'),
            ('NVDA', 'NVIDIA Corporation', 'Technology'),
            ('JPM', 'JPMorgan Chase & Co.', 'Financial'),
            ('JNJ', 'Johnson & Johnson', 'Healthcare'),
            ('V', 'Visa Inc.', 'Financial'),
            ('PG', 'Procter & Gamble Company', 'Consumer Staples'),
            ('UNH', 'UnitedHealth Group Incorporated', 'Healthcare'),
            ('HD', 'Home Depot Inc.', 'Consumer Discretionary'),
            ('MA', 'Mastercard Incorporated', 'Financial'),
            ('BAC', 'Bank of America Corporation', 'Financial'),
            ('XOM', 'Exxon Mobil Corporation', 'Energy'),
            ('PFE', 'Pfizer Inc.', 'Healthcare'),
            ('KO', 'Coca-Cola Company', 'Consumer Staples'),
            ('ABBV', 'AbbVie Inc.', 'Healthcare'),
            ('PEP', 'PepsiCo Inc.', 'Consumer Staples'),
            ('COST', 'Costco Wholesale Corporation', 'Consumer Staples'),
            ('WMT', 'Walmart Inc.', 'Consumer Staples'),
            ('DIS', 'Walt Disney Company', 'Communication'),
            ('TMO', 'Thermo Fisher Scientific Inc.', 'Healthcare'),
            ('VZ', 'Verizon Communications Inc.', 'Communication'),
            ('ADBE', 'Adobe Inc.', 'Technology'),
            ('NFLX', 'Netflix Inc.', 'Communication'),
            ('CRM', 'Salesforce Inc.', 'Technology'),
            ('NEE', 'NextEra Energy Inc.', 'Utilities'),
            ('CMCSA', 'Comcast Corporation', 'Communication'),
            ('T', 'AT&T Inc.', 'Communication'),
            ('ACN', 'Accenture plc', 'Technology'),
            ('LLY', 'Eli Lilly and Company', 'Healthcare'),
            ('WFC', 'Wells Fargo & Company', 'Financial'),
            ('ABT', 'Abbott Laboratories', 'Healthcare'),
            ('CVX', 'Chevron Corporation', 'Energy'),
            ('AMD', 'Advanced Micro Devices Inc.', 'Technology'),
            ('BMY', 'Bristol Myers Squibb Company', 'Healthcare'),
            ('ORCL', 'Oracle Corporation', 'Technology'),
            ('MRK', 'Merck & Co. Inc.', 'Healthcare'),
            ('DHR', 'Danaher Corporation', 'Healthcare'),
            ('INTC', 'Intel Corporation', 'Technology'),
            ('PM', 'Philip Morris International Inc.', 'Consumer Staples'),
            ('LIN', 'Linde plc', 'Materials'),
            ('HON', 'Honeywell International Inc.', 'Industrial'),
            ('UPS', 'United Parcel Service Inc.', 'Industrial'),
            ('QCOM', 'QUALCOMM Incorporated', 'Technology'),
            ('LOW', 'Lowe\'s Companies Inc.', 'Consumer Discretionary')
        ]
        
        df = pd.DataFrame(fallback_data, columns=['Symbol', 'Security', 'Sector'])
        df['GICS_Sector'] = df['Sector']
        df['GICS_Sub_Industry'] = 'Various'
        
        self.sp500_data = df
        self._create_sector_mapping()
        return df
    
    def _create_sector_mapping(self):
        """Create mapping from ticker to sector"""
        if self.sp500_data is not None:
            self.sector_mapping = dict(zip(self.sp500_data['Symbol'], self.sp500_data['Sector']))
    
    def get_sector_for_ticker(self, ticker: str) -> str:
        """Get sector for a specific ticker"""
        return self.sector_mapping.get(ticker, 'Unknown')
    
    def get_tickers_by_sector(self, sector: str) -> List[str]:
        """Get all tickers in a specific sector"""
        if self.sp500_data is None:
            return []
        
        sector_stocks = self.sp500_data[self.sp500_data['Sector'] == sector]
        return sector_stocks['Symbol'].tolist()
    
    def get_all_sectors(self) -> List[str]:
        """Get list of all sectors"""
        if self.sp500_data is None:
            return []
        
        return sorted(self.sp500_data['Sector'].unique())
    
    def get_sector_summary(self) -> Dict[str, int]:
        """Get count of stocks per sector"""
        if self.sp500_data is None:
            return {}
        
        return self.sp500_data['Sector'].value_counts().to_dict()
    
    def get_all_tickers(self) -> List[str]:
        """Get all S&P 500 tickers"""
        if self.sp500_data is None:
            return []
        
        return self.sp500_data['Symbol'].tolist()
    
    def get_company_info(self, ticker: str) -> Dict:
        """Get detailed company information"""
        if self.sp500_data is None:
            return {'name': 'Unknown', 'sector': 'Unknown', 'industry': 'Unknown'}
        
        company_data = self.sp500_data[self.sp500_data['Symbol'] == ticker]
        if company_data.empty:
            return {'name': 'Unknown', 'sector': 'Unknown', 'industry': 'Unknown'}
        
        row = company_data.iloc[0]
        return {
            'name': row.get('Security', 'Unknown'),
            'sector': row.get('Sector', 'Unknown'),
            'industry': row.get('GICS_Sub_Industry', 'Unknown')
        }

# Global instance
sector_manager = SectorDataManager()