// Favorites management
let favorites = [];

// Function to toggle favorite status
function toggleFavorite(ticker, assetType = 'stock', buttonElement = null) {
    fetch('/api/toggle_favorite', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            ticker: ticker,
            asset_type: assetType,
            action: favorites.includes(ticker) ? 'remove' : 'add'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update local favorites array
            if (favorites.includes(ticker)) {
                favorites = favorites.filter(t => t !== ticker);
            } else {
                favorites.push(ticker);
            }

            // Update button appearance if provided
            if (buttonElement) {
                updateFavoriteButton(buttonElement, ticker);
            } else {
                // Update all buttons for this ticker
                updateAllFavoriteButtons(ticker);
            }
        } else {
            console.error('Failed to toggle favorite:', data.error);
        }
    })
    .catch(error => {
        console.error('Error toggling favorite:', error);
    });
}

// Function to update a single favorite button
function updateFavoriteButton(button, ticker) {
    const isFavorite = favorites.includes(ticker);

    if (isFavorite) {
        button.classList.add('favorited');
        button.style.color = '#ffc107';
    } else {
        button.classList.remove('favorited');
        button.style.color = '';
    }

    const icon = button.querySelector('i');
    if (icon) {
        icon.setAttribute('data-feather', 'star');
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }
}

// Function to update all favorite buttons for a ticker
function updateAllFavoriteButtons(ticker) {
    document.querySelectorAll('.favorite-btn').forEach(btn => {
        const onclickAttr = btn.getAttribute('onclick');
        if (onclickAttr && onclickAttr.includes(`'${ticker}'`)) {
            updateFavoriteButton(btn, ticker);
        }
    });
}

// Function to load favorite statuses from server
function loadFavoriteStatuses() {
    fetch('/api/get_favorites')
        .then(response => response.json())
        .then(data => {
            favorites = data.favorites || [];

            // Update favorite button states
            document.querySelectorAll('.favorite-btn').forEach(btn => {
                const ticker = btn.getAttribute('data-ticker');
                if (ticker) {
                    const isFavorite = favorites.includes(ticker);
                    btn.classList.toggle('favorited', isFavorite);
                    btn.innerHTML = isFavorite ? '★' : '☆';
                    btn.title = isFavorite ? 'Remove from favorites' : 'Add to favorites';
                }
            });
        })
        .catch(error => {
            console.error('Error loading favorites:', error);
        });
}

// Load favorite statuses on page load
document.addEventListener('DOMContentLoaded', function() {
    loadFavoriteStatuses();
});

// Export functions for global use
window.toggleFavorite = toggleFavorite;
window.loadFavoriteStatuses = loadFavoriteStatuses;