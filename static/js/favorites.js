
// Favorites management
let favorites = [];

function loadFavoriteStatuses() {
    // Load favorites from API
    fetch('/api/get_favorites')
        .then(response => response.json())
        .then(data => {
            favorites = data.favorites || [];
            updateFavoriteButtons();
        })
        .catch(error => {
            console.error('Error loading favorites:', error);
            favorites = [];
        });
}

function updateFavoriteButtons() {
    // Update all favorite buttons on the page
    document.querySelectorAll('.favorite-btn').forEach(btn => {
        const ticker = btn.getAttribute('data-ticker');
        const isFavorite = favorites.includes(ticker);
        
        btn.classList.toggle('favorited', isFavorite);
        btn.innerHTML = isFavorite ? '★' : '☆';
        btn.title = isFavorite ? 'Remove from favorites' : 'Add to favorites';
    });
}

function toggleFavorite(ticker, assetType = 'stock') {
    const isFavorite = favorites.includes(ticker);
    const action = isFavorite ? 'remove' : 'add';
    
    fetch('/api/toggle_favorite', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            ticker: ticker,
            asset_type: assetType,
            action: action
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (action === 'add') {
                favorites.push(ticker);
            } else {
                favorites = favorites.filter(t => t !== ticker);
            }
            updateFavoriteButtons();
        } else {
            console.error('Error toggling favorite:', data.error);
        }
    })
    .catch(error => {
        console.error('Error toggling favorite:', error);
    });
}

// Initialize favorites when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadFavoriteStatuses();
});
