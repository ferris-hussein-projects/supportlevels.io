// Favorites functionality
class FavoriteManager {
    constructor() {
        this.favorites = new Set();
        this.loadFavorites();
    }

    async loadFavorites() {
        try {
            const response = await fetch('/api/get_favorites');
            const data = await response.json();
            this.favorites = new Set(data.favorites || []);
            this.updateFavoriteButtons();
        } catch (error) {
            console.error('Error loading favorites:', error);
        }
    }

    async toggleFavorite(ticker, assetType = 'stock') {
        const isFavorite = this.favorites.has(ticker);
        const action = isFavorite ? 'remove' : 'add';

        try {
            const response = await fetch('/api/toggle_favorite', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ticker: ticker,
                    asset_type: assetType,
                    action: action
                })
            });

            const result = await response.json();
            if (result.success) {
                if (action === 'add') {
                    this.favorites.add(ticker);
                } else {
                    this.favorites.delete(ticker);
                }
                this.updateFavoriteButtons();
                return true;
            }
        } catch (error) {
            console.error('Error toggling favorite:', error);
        }
        return false;
    }

    updateFavoriteButtons() {
        document.querySelectorAll('[data-ticker]').forEach(button => {
            const ticker = button.dataset.ticker;
            const icon = button.querySelector('i');
            const isFavorite = this.favorites.has(ticker);

            if (icon) {
                icon.className = isFavorite ? 'fas fa-heart text-danger' : 'far fa-heart text-muted';
            }

            button.title = isFavorite ? `Remove ${ticker} from favorites` : `Add ${ticker} to favorites`;
        });
    }

    isFavorite(ticker) {
        return this.favorites.has(ticker);
    }
}

// Global instance
let favoriteManager;

// Global functions for backward compatibility
function loadFavoriteStatuses() {
    if (favoriteManager) {
        favoriteManager.loadFavorites();
    }
}

function toggleFavorite(ticker, assetType = 'stock') {
    if (favoriteManager) {
        return favoriteManager.toggleFavorite(ticker, assetType);
    }
    return false;
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize favorite manager
    favoriteManager = new FavoriteManager();

    // Set up favorite button click handlers
    document.addEventListener('click', function(e) {
        const favoriteBtn = e.target.closest('[data-ticker]');
        if (favoriteBtn && favoriteBtn.classList.contains('favorite-btn')) {
            e.preventDefault();
            const ticker = favoriteBtn.dataset.ticker;
            const assetType = favoriteBtn.dataset.assetType || 'stock';
            favoriteManager.toggleFavorite(ticker, assetType);
        }
    });
});