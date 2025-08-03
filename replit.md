# Stock Support Tracker

## Overview

Stock Support Tracker is a Flask-based web application that monitors the top 20 Fortune 500 stocks and 10 popular cryptocurrencies, identifying those approaching key technical support levels (moving averages). The application provides real-time analysis of stock and crypto prices relative to their 21-day, 50-day, and 200-day moving averages, with comprehensive sector-based filtering, favoriting functionality, and categorization. It features a responsive dark-themed interface with interactive charts, comparison tools, multi-asset filtering, favorites management, and customizable threshold settings with user preferences persistence.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Bootstrap 5 with dark theme for responsive UI
- **Templating**: Jinja2 templates with base template inheritance
- **Visualization**: Chart.js for interactive stock price charts and technical indicators
- **Icons**: Feather Icons for consistent iconography
- **Styling**: Custom CSS overlay on Bootstrap for enhanced card hover effects and chart styling

### Backend Architecture
- **Framework**: Flask web framework with session management
- **Application Structure**: Modular design with separate StockAnalyzer class for business logic
- **Data Processing**: Pandas for data manipulation and NumPy for numerical calculations
- **Stock Data**: yfinance library for real-time market data fetching
- **Technical Analysis**: Custom implementations of RSI, MACD, and moving average calculations
- **Error Handling**: Comprehensive logging and fallback mechanisms for data availability

### Data Management
- **Stock Universe**: Top 20 Fortune 500 companies by market cap (Apple, Microsoft, Google, Amazon, Tesla, Meta, NVIDIA, etc.) with sector classification
- **Cryptocurrency Universe**: Top 10 popular cryptocurrencies (Bitcoin, Ethereum, BNB, XRP, Cardano, Dogecoin, etc.) with category classification
- **Database Storage**: PostgreSQL database for user settings persistence, asset popularity tracking, and favorites management
- **Asset Classification**: 6-sector Fortune 500 categorization for stocks (Technology, Financial Services, Healthcare, etc.) plus crypto categories (Store of Value, Smart Contract, Meme Coin, etc.)
- **Data Sources**: Yahoo Finance API via yfinance for historical and real-time stock and cryptocurrency data
- **User Persistence**: Settings include threshold, sorting preferences, asset filter defaults, and favorites (stocks and crypto)
- **Favorites System**: Session-based favoriting for both stocks and cryptocurrencies with filtering capability

### Key Features
- **Support Level Detection**: Configurable threshold-based detection of stocks and crypto near historical support levels (1M, 6M, 1Y, 5Y)
- **Multi-Asset Filtering**: Filter by Fortune 500 sectors (6 categories) or crypto categories (6 types) with visual count indicators
- **Favorites Management**: Add/remove stocks and cryptocurrencies to favorites with dedicated filtering and session persistence
- **Comprehensive Data**: Top 20 Fortune 500 stocks plus 10 popular cryptocurrencies with detailed information
- **Unified Analysis**: Stocks and cryptocurrency support analysis in single interface with asset type indicators
- **Multi-Asset Comparison**: Side-by-side analysis of up to 5 stocks or cryptocurrencies
- **Technical Indicators**: RSI, MACD, volatility calculations, and multiple moving average calculations for both asset types
- **Popularity Tracking**: Database-based view count tracking for stock and crypto analysis pages
- **User Settings Persistence**: PostgreSQL storage for threshold, sorting, asset filter preferences, and favorites
- **Progressive Web App**: PWA capabilities with manifest.json and service worker for mobile installation
- **Market Dashboard**: Overview of major market indices and cryptocurrency performance
- **Export Functionality**: CSV export of analysis results for both stocks and crypto
- **Responsive Design**: Mobile-friendly interface with collapsible navigation and asset type badges
- **Health Check Endpoint**: Dedicated `/health` endpoint for deployment monitoring and fast response times

## External Dependencies

### Market Data APIs
- **Yahoo Finance**: Primary data source via yfinance Python library for stock prices, historical data, and company information
- **Wikipedia**: S&P 500 ticker list scraping for dynamic stock universe updates

### Frontend Libraries
- **Bootstrap 5**: UI framework with Replit dark theme customization
- **Chart.js**: JavaScript charting library for price charts and technical indicator visualization
- **Feather Icons**: SVG icon library for consistent iconography

### Python Libraries
- **Flask**: Web framework for application structure and routing
- **Flask-SQLAlchemy**: ORM for PostgreSQL database operations
- **yfinance**: Yahoo Finance API wrapper for market data
- **pandas**: Data manipulation and analysis for S&P 500 sector data
- **numpy**: Numerical computing for technical analysis calculations
- **requests**: HTTP requests for web scraping S&P 500 data
- **beautifulsoup4**: HTML parsing for sector classification data

### Hosting Environment
- **Replit**: Cloud development and hosting platform
- **PostgreSQL Database**: Managed database service for user settings and analytics
- **Environment Variables**: SESSION_SECRET for Flask session security, DATABASE_URL for PostgreSQL connection