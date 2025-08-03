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

// Function to load favorite statuses from server
function loadFavoriteStatuses() {
    fetch('/api/get_favorites')
        .then(response => response.json())
        .then(data => {
            const favorites = data.favorites || [];

            // Update all favorite buttons
            document.querySelectorAll('.favorite-btn').forEach(btn => {
                try {
                    const onclickAttr = btn.getAttribute('onclick');
                    if (onclickAttr) {
                        const tickerMatch = onclickAttr.match(/'([^']+)'/);
                        if (tickerMatch) {
                            const ticker = tickerMatch[1];
                            if (favorites.includes(ticker)) {
                                btn.classList.add('favorited');
                                const icon = btn.querySelector('i');
                                if (icon) {
                                    icon.setAttribute('data-feather', 'star');
                                }
                            } else {
                                btn.classList.remove('favorited');
                                const icon = btn.querySelector('i');
                                if (icon) {
                                    icon.setAttribute('data-feather', 'star');
                                }
                            }
                        }
                    }
                } catch (error) {
                    console.error('Error processing favorite button:', error);
                }
            });

            // Replace feather icons
            if (typeof feather !== 'undefined') {
                feather.replace();
            }
        })
        .catch(error => {
            console.error('Error loading favorites:', error);
        });
}

// Load favorite statuses on page load
document.addEventListener('DOMContentLoaded', function() {
    loadFavoriteStatuses();
});