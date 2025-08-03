from app import db
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @classmethod
    def create_default_users(cls):
        """Create default users if they don't exist"""
        # Create admin user
        admin = cls.query.filter_by(username='hussef01').first()
        if not admin:
            admin = cls(username='hussef01', is_admin=True)
            admin.set_password('01hussef')
            db.session.add(admin)

        # Create demo user
        demo = cls.query.filter_by(username='demo').first()
        if not demo:
            demo = cls(username='demo', is_admin=False)
            demo.set_password('demo123')
            db.session.add(demo)

        db.session.commit()


class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(255), unique=True, nullable=False)
    threshold = db.Column(db.Float, default=3.0)  # Threshold percentage
    sort_by = db.Column(db.String(50), default='ticker')  # ticker, price, distance, popularity
    sort_order = db.Column(db.String(10), default='asc')  # asc, desc
    sector_filter = db.Column(db.String(100), default='All')  # sector filter preference
    selected_tickers = db.Column(db.Text)  # JSON string of selected tickers
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_selected_tickers(self):
        if self.selected_tickers:
            return json.loads(self.selected_tickers)
        return []

    def set_selected_tickers(self, tickers):
        self.selected_tickers = json.dumps(tickers)


class StockPopularity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    view_count = db.Column(db.Integer, default=0)
    last_viewed = db.Column(db.DateTime, default=datetime.utcnow)

    @classmethod
    def increment_view(cls, ticker):
        stock = cls.query.filter_by(ticker=ticker).first()
        if stock:
            stock.view_count += 1
            stock.last_viewed = datetime.utcnow()
        else:
            stock = cls(ticker=ticker, view_count=1)
            db.session.add(stock)
        db.session.commit()
        return stock

    @classmethod
    def get_popularity_score(cls, ticker):
        stock = cls.query.filter_by(ticker=ticker).first()
        return stock.view_count if stock else 0


class UserFavorites(db.Model):
    __tablename__ = 'user_favorites'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(255), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    asset_type = db.Column(db.String(10), nullable=False)  # 'stock' or 'crypto'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('session_id', 'ticker', 'asset_type', name='unique_session_ticker_asset'),
    )

    @staticmethod
    def get_session_id():
        """Get or create session ID for guest users"""
        from flask import session
        if 'user_session_id' not in session:
            import uuid
            session['user_session_id'] = str(uuid.uuid4())
        return session['user_session_id']

    @staticmethod
    def add_favorite(ticker, asset_type):
        """Add a ticker to favorites"""
        session_id = UserFavorites.get_session_id()
        existing = UserFavorites.query.filter_by(
            session_id=session_id,
            ticker=ticker.upper(),
            asset_type=asset_type
        ).first()

        if not existing:
            favorite = UserFavorites(
                session_id=session_id,
                ticker=ticker.upper(),
                asset_type=asset_type
            )
            db.session.add(favorite)
            db.session.commit()
            return True
        return False

    @staticmethod
    def remove_favorite(ticker, asset_type):
        """Remove a ticker from favorites"""
        session_id = UserFavorites.get_session_id()
        favorite = UserFavorites.query.filter_by(
            session_id=session_id,
            ticker=ticker.upper(),
            asset_type=asset_type
        ).first()

        if favorite:
            db.session.delete(favorite)
            db.session.commit()
            return True
        return False

    @staticmethod
    def get_favorites():
        """Get all favorites for current user/session"""
        session_id = UserFavorites.get_session_id()
        favorites = UserFavorites.query.filter_by(session_id=session_id).all()
        return [{'ticker': fav.ticker, 'asset_type': fav.asset_type} for fav in favorites]

    @staticmethod
    def get_favorite_tickers():
        """Get list of favorite ticker symbols"""
        favorites = UserFavorites.get_favorites()
        return [fav['ticker'] for fav in favorites]


class TopAssetConfiguration(db.Model):
    """Store configurable top stocks and crypto lists"""
    __tablename__ = 'top_asset_configuration'

    id = db.Column(db.Integer, primary_key=True)
    asset_type = db.Column(db.String(10), nullable=False)  # 'stock' or 'crypto'
    asset_list = db.Column(db.Text, nullable=False)  # JSON string of asset symbols
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(db.String(255), default='system')  # Track who made changes

    __table_args__ = (
        db.UniqueConstraint('asset_type', 'is_active', name='unique_active_asset_type'),
    )

    def get_asset_list(self):
        """Get asset list as Python list"""
        if self.asset_list:
            return json.loads(self.asset_list)
        return []

    def set_asset_list(self, assets):
        """Set asset list from Python list"""
        self.asset_list = json.dumps(assets)
        self.updated_at = datetime.utcnow()

    @classmethod
    def get_top_stocks(cls):
        """Get current active top stocks list"""
        config = cls.query.filter_by(asset_type='stock', is_active=True).first()
        if config:
            return config.get_asset_list()
        # Return default if no config exists
        from stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        return analyzer.DEFAULT_TOP_STOCKS

    @classmethod
    def get_top_crypto(cls):
        """Get current active top crypto list"""
        config = cls.query.filter_by(asset_type='crypto', is_active=True).first()
        if config:
            return config.get_asset_list()
        # Return default if no config exists
        from crypto_data import crypto_manager
        return crypto_manager.DEFAULT_TOP_CRYPTO

    @classmethod
    def set_top_stocks(cls, stock_list, updated_by='admin'):
        """Set new top stocks configuration"""
        try:
            # Deactivate existing config
            existing = cls.query.filter_by(asset_type='stock', is_active=True).first()
            if existing:
                existing.is_active = False

            # Create new config
            new_config = cls(
                asset_type='stock',
                is_active=True,
                updated_by=updated_by
            )
            new_config.set_asset_list(stock_list)

            db.session.add(new_config)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            return False

    @classmethod
    def set_top_crypto(cls, crypto_list, updated_by='admin'):
        """Set new top crypto configuration"""
        try:
            # Deactivate existing config
            existing = cls.query.filter_by(asset_type='crypto', is_active=True).first()
            if existing:
                existing.is_active = False

            # Create new config
            new_config = cls(
                asset_type='crypto',
                is_active=True,
                updated_by=updated_by
            )
            new_config.set_asset_list(crypto_list)

            db.session.add(new_config)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            return False

    @classmethod
    def get_configuration_history(cls, asset_type, limit=10):
        """Get configuration change history"""
        return cls.query.filter_by(asset_type=asset_type)\
                       .order_by(cls.updated_at.desc())\
                       .limit(limit).all()