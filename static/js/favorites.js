
// Favorites functionality
class FavoriteManager {
    constructor() {
        this.favorites = new Set();
        this.isInitialized = false;
        this.loadFavorites();
    }

    async loadFavorites() {
        try {
            const response = await fetch('/api/get_favorites');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            this.favorites = new Set(data.favorites || []);
            this.isInitialized = true;
            this.updateFavoriteButtons();
            console.log('Favorites loaded:', this.favorites);
        } catch (error) {
            console.error('Error loading favorites:', error);
            this.favorites = new Set(); // Ensure it's initialized even on error
            this.isInitialized = true;
        }
    }

    async toggleFavorite(ticker, assetType = 'stock') {
        if (!this.isInitialized) {
            console.log('Favorites not initialized yet, waiting...');
            await this.loadFavorites();
        }

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

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                if (action === 'add') {
                    this.favorites.add(ticker);
                } else {
                    this.favorites.delete(ticker);
                }
                this.updateFavoriteButtons();
                console.log(`${action}ed ${ticker} to favorites`);
                return true;
            } else {
                console.error('Server error:', result.error);
            }
        } catch (error) {
            console.error('Error toggling favorite:', error);
        }
        return false;
    }

    updateFavoriteButtons() {
        if (!this.isInitialized) {
            return;
        }

        document.querySelectorAll('[data-ticker]').forEach(button => {
            const ticker = button.dataset.ticker;
            if (!ticker) return;

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
let favoriteManager = null;

// Global functions for backward compatibility
function loadFavoriteStatuses() {
    if (favoriteManager) {
        favoriteManager.loadFavorites();
    } else {
        console.log('FavoriteManager not initialized yet');
    }
}

function toggleFavorite(ticker, assetType = 'stock') {
    if (favoriteManager) {
        return favoriteManager.toggleFavorite(ticker, assetType);
    } else {
        console.log('FavoriteManager not initialized yet');
        return Promise.resolve(false);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing FavoriteManager');
    
    // Initialize favorite manager
    favoriteManager = new FavoriteManager();

    // Set up favorite button click handlers
    document.addEventListener('click', function(e) {
        const favoriteBtn = e.target.closest('[data-ticker]');
        if (favoriteBtn && (favoriteBtn.classList.contains('favorite-btn') || favoriteBtn.getAttribute('onclick'))) {
            e.preventDefault();
            const ticker = favoriteBtn.dataset.ticker;
            const assetType = favoriteBtn.dataset.assetType || 'stock';
            
            if (favoriteManager) {
                favoriteManager.toggleFavorite(ticker, assetType);
            }
        }
    });

    // Also handle any existing onclick handlers by updating buttons periodically
    setInterval(() => {
        if (favoriteManager && favoriteManager.isInitialized) {
            favoriteManager.updateFavoriteButtons();
        }
    }, 1000);
});

// Ensure favorites are loaded when called from templates
window.loadFavoriteStatuses = loadFavoriteStatuses;
window.toggleFavorite = toggleFavorite;
window.favoriteManager = favoriteManager;
