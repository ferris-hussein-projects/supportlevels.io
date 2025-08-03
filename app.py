import os
import logging
from flask import Flask, render_template, request, redirect, url_for, session, make_response, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
from stock_analyzer import StockAnalyzer
import uuid
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Import models and initialize db
from models import db, User, UserSettings, StockPopularity, TopAssetConfiguration
db.init_app(app)

with app.app_context():
    db.create_all()
    # Create default users
    User.create_default_users()

# Initialize stock analyzer
analyzer = StockAnalyzer()

def sync_analyzer_with_db():
    """Sync analyzer and crypto manager with database configurations"""
    try:
        # Sync stock analyzer with database
        top_stocks = TopAssetConfiguration.get_top_stocks()
        analyzer.set_top_stocks(top_stocks)
        
        # Sync crypto manager with database
        from crypto_data import crypto_manager
        top_crypto = TopAssetConfiguration.get_top_crypto()
        crypto_manager.set_top_crypto(top_crypto)
        
        logging.info(f"Synced with DB: {len(top_stocks)} top stocks, {len(top_crypto)} top crypto")
    except Exception as e:
        logging.error(f"Error syncing with database: {e}")

# Sync with database on startup - within app context
with app.app_context():
    sync_analyzer_with_db()

def get_or_create_session_id():
    """Get or create a unique session ID for settings persistence"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash('Admin access required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current logged in user"""
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

def get_user_settings():
    """Get user settings from database"""
    session_id = get_or_create_session_id()
    settings = UserSettings.query.filter_by(session_id=session_id).first()
    if not settings:
        settings = UserSettings(session_id=session_id)
        db.session.add(settings)
        db.session.commit()
    return settings

def sort_stocks(results, sort_by='ticker', sort_order='asc'):
    """Sort stock results by specified criteria"""
    if not results:
        return results
    
    # Add popularity scores to results
    for result in results:
        try:
            if result and isinstance(result, dict) and 'ticker' in result:
                result['popularity'] = StockPopularity.get_popularity_score(result['ticker'])
                # Calculate distance from nearest support level
                try:
                    price = float(result.get('price', 0))
                    distances = []
                    if result.get('ma21'):
                        distances.append(abs(price - float(result['ma21'])) / float(result['ma21']))
                    if result.get('ma50'):
                        distances.append(abs(price - float(result['ma50'])) / float(result['ma50']))
                    if result.get('ma200'):
                        distances.append(abs(price - float(result['ma200'])) / float(result['ma200']))
                    result['distance'] = min(distances) if distances else 0
                except (ValueError, TypeError):
                    result['distance'] = 0
            else:
                # Skip invalid results
                logging.warning(f"Invalid result in sort_stocks: {result}")
                continue
        except Exception as e:
            logging.error(f"Error processing result in sort_stocks: {e}")
            continue
    
    # Filter out any None or invalid results
    valid_results = [r for r in results if r and isinstance(r, dict) and 'ticker' in r]
    
    # Define sort key
    if sort_by == 'price':
        key_func = lambda x: float(x.get('price', 0))
    elif sort_by == 'popularity':
        key_func = lambda x: x.get('popularity', 0)
    elif sort_by == 'distance':
        key_func = lambda x: x.get('distance', 0)
    else:  # ticker
        key_func = lambda x: x.get('ticker', '')
    
    reverse = sort_order == 'desc'
    return sorted(valid_results, key=key_func, reverse=reverse)



@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            flash(f'Welcome, {user.username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Main page showing stocks approaching support or resistance levels"""
    try:
        settings = get_user_settings()
        support_threshold = settings.support_threshold / 100.0  # Convert percentage to decimal
        resistance_threshold = settings.resistance_threshold / 100.0  # Convert percentage to decimal
        level_type = request.args.get('level_type', settings.level_type or 'support')
        sort_by = request.args.get('sort', settings.sort_by)
        sort_order = request.args.get('order', settings.sort_order)
        sector_filter = request.args.get('sector', settings.sector_filter or 'All')
        
        # Import managers here to avoid circular imports
        from crypto_data import crypto_manager
        
        # Get stocks and crypto near support/resistance with filtering
        results = []
        try:
            results = analyzer.get_stocks_near_levels(support_threshold, resistance_threshold, level_type, sector_filter, include_crypto=True)
            if not results:
                results = []
        except Exception as data_error:
            logging.error(f"Error fetching stock data: {data_error}")
            results = []
        
        # Debug logging
        current_threshold = support_threshold if level_type == 'support' else resistance_threshold
        logging.info(f"Level type: {level_type}, Threshold: {current_threshold}, Sector filter: {sector_filter}")
        logging.info(f"Raw results count: {len(results)}")
        
        # Sort results safely
        try:
            if results:  # Only sort if we have results
                results = sort_stocks(results, sort_by, sort_order)
        except Exception as sort_error:
            logging.error(f"Error sorting results: {sort_error}")
            results = []
        
        # Get summary statistics with error handling
        try:
            total_stocks_available = len(analyzer.get_all_sp500_tickers())
            total_stocks_tracked = len(analyzer.get_top_stocks())
            total_crypto_available = len(crypto_manager.get_all_crypto_symbols())
            total_crypto_tracked = len(crypto_manager.get_top_crypto())
        except Exception as stats_error:
            logging.error(f"Error getting statistics: {stats_error}")
            total_stocks_available = 0
            total_stocks_tracked = 0
            total_crypto_available = 0
            total_crypto_tracked = 0
        
        total_assets_tracked = total_stocks_tracked + total_crypto_tracked
        assets_near_levels = len(results)
        
        # Get sector information for filtering (conservative groupings)
        stock_sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Industrial', 'Energy', 'Communication', 'Utilities']
        crypto_categories = ['Crypto']  # Simplified to just "Crypto"
        all_filters = stock_sectors + crypto_categories
        
        # Combine sector summaries with error handling
        sector_summary = {}
        try:
            for sector in stock_sectors:
                count = sum(1 for ticker in analyzer.TICKERS if analyzer.get_stock_sector(ticker) == sector)
                sector_summary[sector] = count
            
            # Simplified crypto summary (all crypto under "Crypto")
            sector_summary['Crypto'] = len(crypto_manager.get_top_crypto())
        except Exception as sector_error:
            logging.error(f"Error calculating sector summary: {sector_error}")
            sector_summary = {'All': 0}
        
        # Calculate threshold for template (use support_threshold for display)
        current_threshold = support_threshold if level_type == 'support' else resistance_threshold
        
        return render_template('index.html', 
                             results=results, 
                             support_threshold=support_threshold * 100,  # Convert back to percentage
                             resistance_threshold=resistance_threshold * 100,  # Convert back to percentage
                             threshold=current_threshold * 100,  # Add threshold for template compatibility
                             level_type=level_type,
                             total_stocks_available=total_stocks_available,
                             total_stocks_tracked=total_stocks_tracked,
                             total_crypto_available=total_crypto_available,
                             total_crypto_tracked=total_crypto_tracked,
                             total_assets_tracked=total_assets_tracked,
                             assets_near_levels=assets_near_levels,
                             sort_by=sort_by,
                             sort_order=sort_order,
                             sector_filter=sector_filter,
                             all_sectors=all_filters,
                             sector_summary=sector_summary,
                             total_stocks=total_stocks_tracked,
                             total_assets=total_assets_tracked)
    except Exception as e:
        logging.error(f"Critical error in index route: {e}")
        return render_template('index.html', 
                             results=[], 
                             support_threshold=3.0,
                             resistance_threshold=3.0,
                             threshold=3.0,  # Add threshold for template compatibility
                             level_type='support',
                             total_stocks_available=0,
                             total_stocks_tracked=0,
                             total_crypto_available=0,
                             total_crypto_tracked=0,
                             total_assets_tracked=0,
                             assets_near_levels=0,
                             sort_by='ticker',
                             sort_order='asc',
                             sector_filter='All',
                             all_sectors=[],
                             sector_summary={},
                             total_stocks=0,
                             total_assets=0,
                             error="Unable to fetch stock data. Please try again later.")

@app.route('/landing')
def landing():
    """Simple landing page without expensive operations for health checks"""
    return render_template('landing.html')

@app.route('/health')
def health_check():
    """Simple health check endpoint for deployment"""
    from datetime import datetime
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

@app.route('/debug')
@login_required  
def debug():
    """Debug endpoint to check analyzer status"""
    try:
        settings = get_user_settings()
        threshold = settings.threshold / 100.0
        
        # Get basic analyzer info
        top_stocks = analyzer.get_top_stocks()
        all_results = analyzer.get_stocks_near_support(0.10, 'All', include_crypto=True)  # 10% threshold
        
        debug_info = {
            'threshold_setting': settings.threshold,
            'threshold_decimal': threshold,
            'top_stocks_count': len(top_stocks),
            'top_stocks_sample': top_stocks[:5],
            'all_results_count': len(all_results) if all_results else 0,
            'results_sample': all_results[:3] if all_results else [],
            'analyzer_tickers_count': len(analyzer.TICKERS) if hasattr(analyzer, 'TICKERS') else 0
        }
        
        return debug_info
    except Exception as e:
        return {'error': str(e)}

@app.route('/analysis')
@login_required
def analysis():
    """Main page showing stocks approaching support levels"""
    settings = get_user_settings()
    threshold = settings.threshold / 100.0  # Convert percentage to decimal
    sort_by = request.args.get('sort', settings.sort_by)
    sort_order = request.args.get('order', settings.sort_order)
    sector_filter = request.args.get('sector', settings.sector_filter or 'All')
    
    try:
        # Import managers here to avoid circular imports
        from sector_data import sector_manager
        from crypto_data import crypto_manager
        
        # Get stocks and crypto near support with filtering
        results = analyzer.get_stocks_near_support(threshold, sector_filter, include_crypto=True)
        
        # Note: Removed halal filtering - show all assets with halal indicators
        
        # Sort results
        results = sort_stocks(results, sort_by, sort_order)
        
        # Get summary statistics
        total_stocks_available = len(analyzer.get_all_sp500_tickers())
        total_stocks_tracked = len(analyzer.get_top_stocks())
        total_crypto_available = len(crypto_manager.get_all_crypto_symbols())
        total_crypto_tracked = len(crypto_manager.get_top_crypto())
        total_assets_tracked = total_stocks_tracked + total_crypto_tracked
        assets_near_support = len(results)
        
        # Get sector information for filtering (conservative groupings)
        stock_sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Industrial', 'Energy', 'Communication', 'Utilities']
        crypto_categories = ['Crypto']  # Simplified to just "Crypto"
        all_filters = stock_sectors + crypto_categories
        
        # Combine sector summaries
        sector_summary = {}
        for sector in stock_sectors:
            count = sum(1 for ticker in analyzer.TICKERS if analyzer.get_stock_sector(ticker) == sector)
            sector_summary[sector] = count
        
        # Simplified crypto summary (all crypto under "Crypto")
        sector_summary['Crypto'] = len(crypto_manager.get_top_crypto())
        
        return render_template('analysis.html', 
                             results=results, 
                             threshold=threshold * 100,  # Convert back to percentage
                             total_stocks_available=total_stocks_available,
                             total_stocks_tracked=total_stocks_tracked,
                             total_crypto_available=total_crypto_available,
                             total_crypto_tracked=total_crypto_tracked,
                             total_assets_tracked=total_assets_tracked,
                             assets_near_support=assets_near_support,
                             sort_by=sort_by,
                             sort_order=sort_order,
                             sector_filter=sector_filter,
                             all_sectors=all_filters,
                             sector_summary=sector_summary)
    except Exception as e:
        logging.error(f"Error in analysis route: {e}")
        return render_template('analysis.html', 
                             results=[], 
                             threshold=threshold * 100,
                             total_stocks_available=0,
                             total_stocks_tracked=0,
                             total_crypto_available=0,
                             total_crypto_tracked=0,
                             total_assets_tracked=0,
                             assets_near_support=0,
                             sort_by='ticker',
                             sort_order='asc',
                             sector_filter='All',
                             all_sectors=[],
                             sector_summary={},
                             error="Unable to fetch stock data. Please try again later.")

# API endpoints for favorites
@app.route('/api/toggle_favorite', methods=['POST'])
def api_toggle_favorite():
    """Toggle favorite status for a ticker"""
    try:
        from models import UserFavorites
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        asset_type = data.get('asset_type', 'stock')
        action = data.get('action', 'add')
        
        if action == 'add':
            success = UserFavorites.add_favorite(ticker, asset_type)
        else:
            success = UserFavorites.remove_favorite(ticker, asset_type)
        
        return {"success": success}
    except Exception as e:
        logging.error(f"Error toggling favorite: {e}")
        return {"success": False, "error": str(e)}, 500

@app.route('/api/get_favorites')
def api_get_favorites():
    """Get user's favorite tickers"""
    try:
        from models import UserFavorites
        favorites = UserFavorites.get_favorites()
        return {"favorites": [fav['ticker'] for fav in favorites]}
    except Exception as e:
        logging.error(f"Error getting favorites: {e}")
        return {"favorites": []}, 500

@app.route('/stock/<ticker>')
@login_required
def stock_detail(ticker):
    """Detailed analysis page for individual stock or crypto"""
    try:
        # Track popularity
        StockPopularity.increment_view(ticker.upper())
        
        # Check if it's a crypto symbol
        from crypto_data import crypto_manager
        if ticker.upper() in [s.upper() for s in crypto_manager.get_all_crypto_symbols()]:
            # Get detailed crypto analysis
            analysis = crypto_manager.get_detailed_crypto_analysis(ticker.upper())
            if 'error' not in analysis:
                analysis['asset_type'] = 'crypto'
        else:
            # Get detailed stock analysis
            analysis = analyzer.get_detailed_analysis(ticker.upper())
        
        if analysis.get('error'):
            return render_template('stock_detail.html', 
                                 ticker=ticker.upper(),
                                 error=analysis['error'])
        
        return render_template('stock_detail.html', 
                             ticker=ticker.upper(),
                             analysis=analysis)
    except Exception as e:
        logging.error(f"Error in stock_detail route for {ticker}: {e}")
        return render_template('stock_detail.html', 
                             ticker=ticker.upper(),
                             error="Unable to fetch detailed stock data. Please try again later.")

@app.route('/search', methods=['POST'])
@login_required
def search_stock():
    """Search for a specific stock"""
    ticker = request.form.get('ticker', '').upper().strip()
    if ticker:
        return redirect(url_for('stock_detail', ticker=ticker))
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard with summary statistics and market overview"""
    try:
        threshold = float(session.get('threshold', 0.03))
        
        # Get dashboard data
        dashboard_data = analyzer.get_dashboard_data(threshold)
        
        return render_template('dashboard.html', 
                             data=dashboard_data,
                             threshold=threshold)
    except Exception as e:
        logging.error(f"Error in dashboard route: {e}")
        return render_template('dashboard.html', 
                             error="Unable to fetch dashboard data. Please try again later.")

@app.route('/compare')
@login_required
def compare():
    """Stock comparison page"""
    tickers = request.args.getlist('tickers')
    comparison_data = None
    
    if tickers and len(tickers) >= 2:
        try:
            # Limit to 5 stocks for performance
            tickers = tickers[:5]
            comparison_data = analyzer.compare_stocks(tickers)
        except Exception as e:
            logging.error(f"Error comparing stocks {tickers}: {e}")
            comparison_data = {'error': 'Unable to compare selected stocks. Please try again.'}
    
    return render_template('compare.html', 
                         available_tickers=analyzer.TICKERS[:100],  # Limit for dropdown
                         selected_tickers=tickers,
                         comparison_data=comparison_data)

@app.route('/admin')
@admin_required
def admin():
    """Admin dashboard for managing top assets"""
    try:
        # Get current configurations
        top_stocks = TopAssetConfiguration.get_top_stocks()
        top_crypto = TopAssetConfiguration.get_top_crypto()
        
        # Get available options
        all_sp500 = analyzer.get_all_sp500_tickers()
        from crypto_data import crypto_manager
        all_crypto = crypto_manager.get_all_crypto_symbols()
        
        # Get statistics
        total_sp500 = len(all_sp500)
        total_crypto_available = len(all_crypto)
        
        return render_template('admin.html',
                             top_stocks=top_stocks,
                             top_crypto=top_crypto,
                             all_sp500=all_sp500,
                             all_crypto=all_crypto,
                             total_sp500=total_sp500,
                             total_crypto_available=total_crypto_available)
    except Exception as e:
        logging.error(f"Error in admin route: {e}")
        return render_template('admin.html', error="Unable to load admin data")

@app.route('/admin/update_top_stocks', methods=['POST'])
@admin_required
def admin_update_top_stocks():
    """Update top stocks configuration"""
    try:
        data = request.get_json()
        stock_list = data.get('stocks', [])
        
        # Validate stocks are in S&P 500
        all_sp500 = analyzer.get_all_sp500_tickers()
        invalid_stocks = [s for s in stock_list if s not in all_sp500]
        
        if invalid_stocks:
            return {"success": False, "error": f"Invalid stocks: {invalid_stocks}"}
        
        # Update database
        success = TopAssetConfiguration.set_top_stocks(stock_list, 'admin')
        
        if success:
            # Sync analyzer
            analyzer.set_top_stocks(stock_list)
            return {"success": True, "message": f"Updated top stocks list to {len(stock_list)} stocks"}
        else:
            return {"success": False, "error": "Failed to update database"}
            
    except Exception as e:
        logging.error(f"Error updating top stocks: {e}")
        return {"success": False, "error": str(e)}

@app.route('/admin/update_top_crypto', methods=['POST'])
@admin_required
def admin_update_top_crypto():
    """Update top crypto configuration"""
    try:
        data = request.get_json()
        crypto_list = data.get('crypto', [])
        
        # Validate crypto symbols
        from crypto_data import crypto_manager
        all_crypto = crypto_manager.get_all_crypto_symbols()
        invalid_crypto = [c for c in crypto_list if c not in all_crypto]
        
        if invalid_crypto:
            return {"success": False, "error": f"Invalid crypto: {invalid_crypto}"}
        
        # Update database
        success = TopAssetConfiguration.set_top_crypto(crypto_list, 'admin')
        
        if success:
            # Sync crypto manager
            crypto_manager.set_top_crypto(crypto_list)
            return {"success": True, "message": f"Updated top crypto list to {len(crypto_list)} cryptocurrencies"}
        else:
            return {"success": False, "error": "Failed to update database"}
            
    except Exception as e:
        logging.error(f"Error updating top crypto: {e}")
        return {"success": False, "error": str(e)}

@app.route('/admin/reset_defaults', methods=['POST'])
@admin_required
def admin_reset_defaults():
    """Reset to default configurations"""
    try:
        data = request.get_json()
        asset_type = data.get('asset_type')
        
        if asset_type == 'stocks':
            success = TopAssetConfiguration.set_top_stocks(analyzer.DEFAULT_TOP_STOCKS, 'admin_reset')
            if success:
                analyzer.reset_to_default_top_stocks()
                return {"success": True, "message": "Reset to default top 20 stocks"}
        elif asset_type == 'crypto':
            from crypto_data import crypto_manager
            success = TopAssetConfiguration.set_top_crypto(crypto_manager.DEFAULT_TOP_CRYPTO, 'admin_reset')
            if success:
                crypto_manager.reset_to_default_top_crypto()
                return {"success": True, "message": "Reset to default top 10 crypto"}
        
        return {"success": False, "error": "Invalid asset type or update failed"}
        
    except Exception as e:
        logging.error(f"Error resetting defaults: {e}")
        return {"success": False, "error": str(e)}

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Settings page for configuring thresholds and preferences"""
    settings = get_user_settings()
    
    if request.method == 'POST':
        try:
            # Update settings
            support_threshold = float(request.form.get('support_threshold', 0.1))
            resistance_threshold = float(request.form.get('resistance_threshold', 0.1))
            level_type = request.form.get('level_type', 'support')
            sort_by = request.form.get('sort_by', 'ticker')
            sort_order = request.form.get('sort_order', 'asc')
            sector_filter = request.form.get('sector_filter', 'All')
            
            if 0.1 <= support_threshold <= 50.0 and 0.1 <= resistance_threshold <= 50.0:  # Validate percentage range
                settings.support_threshold = support_threshold
                settings.resistance_threshold = resistance_threshold
                settings.level_type = level_type
                settings.sort_by = sort_by
                settings.sort_order = sort_order
                settings.sector_filter = sector_filter
                settings.updated_at = datetime.utcnow()
                db.session.commit()
                
            return redirect(url_for('settings'))
        except ValueError:
            pass
    
    # Get sector information for the settings page
    stock_sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Industrial', 'Energy', 'Communication', 'Utilities']
    crypto_categories = ['Crypto']  # Simplified crypto category
    all_filters = stock_sectors + crypto_categories
    
    return render_template('settings.html', settings=settings, all_sectors=all_filters)

@app.route('/export/csv')
@login_required
def export_csv():
    """Export current support data as CSV"""
    try:
        settings = get_user_settings()
        threshold = settings.threshold / 100.0
        sector_filter = request.args.get('sector', settings.sector_filter or 'All')
        
        # Get the same results as displayed on the page
        level_type = request.args.get('level_type', 'support')
        results = analyzer.get_stocks_near_levels(threshold, threshold, level_type, sector_filter, include_crypto=True)
        
        # Create CSV content with all support/resistance levels and halal status
        if level_type == 'support':
            csv_content = "Ticker,Company Name,Asset Type,Sector,Price,Support Zones,Support Prices,1M Support,6M Support,1Y Support,5Y Support,Is Halal,Volume\n"
        else:
            csv_content = "Ticker,Company Name,Asset Type,Sector,Price,Resistance Zones,Resistance Prices,1M Resistance,6M Resistance,1Y Resistance,5Y Resistance,Is Halal,Volume\n"
        
        for result in results:
            # Format halal status
            if result.get('asset_type') == 'crypto':
                halal_status = "Yes (Crypto)"
            elif result.get('is_halal'):
                halal_status = "Yes"
            else:
                halal_status = "No"
            
            # Escape quotes in company name and support zones
            company_name = (result.get('company_name', result['ticker']) or result['ticker']).replace('"', '""')
            zones = (result.get('zones', '') or '').replace('"', '""')
            support_prices = (result.get('support_prices', '') or '').replace('"', '""')
            
            if level_type == 'support':
                support_prices = (result.get('support_prices', '') or '').replace('"', '""')
                csv_content += f'"{result["ticker"]}","{company_name}","{result.get("asset_type", "stock").title()}","{result.get("sector", "Other")}",{result["price"]},"{zones}","{support_prices}",'
                csv_content += f'{result.get("support_1m", "")},{result.get("support_6m", "")},{result.get("support_1y", "")},{result.get("support_5y", "")},'
            else:
                resistance_prices = (result.get('resistance_prices', '') or '').replace('"', '""')
                csv_content += f'"{result["ticker"]}","{company_name}","{result.get("asset_type", "stock").title()}","{result.get("sector", "Other")}",{result["price"]},"{zones}","{resistance_prices}",'
                csv_content += f'{result.get("resistance_1m", "")},{result.get("resistance_6m", "")},{result.get("resistance_1y", "")},{result.get("resistance_5y", "")},'
            csv_content += f'"{halal_status}",{result.get("volume", "")}\n'
        
        # Create response
        response = make_response(csv_content)
        response.headers["Content-Disposition"] = f"attachment; filename={level_type}_stocks_{datetime.now().strftime('%Y%m%d')}.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
    except Exception as e:
        logging.error(f"Error exporting CSV: {e}")
        return redirect(url_for('index'))

@app.route('/api/chart_data/<ticker>')
def chart_data(ticker):
    """API endpoint for chart data (exception to avoid fetch calls rule for charts)"""
    try:
        period = request.args.get('period', '6mo')
        chart_data = analyzer.get_chart_data(ticker.upper(), period)
        return json.dumps(chart_data)
    except Exception as e:
        logging.error(f"Error getting chart data for {ticker}: {e}")
        return json.dumps({'error': 'Unable to fetch chart data'})

@app.route('/manifest.json')
def manifest():
    """PWA manifest file"""
    return app.send_static_file('manifest.json')

@app.route('/sw.js')
def service_worker():
    """Service worker for PWA"""
    response = make_response(app.send_static_file('sw.js'))
    response.headers['Content-Type'] = 'application/javascript'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
