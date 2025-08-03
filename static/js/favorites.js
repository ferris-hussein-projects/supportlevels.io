// Favorites functionality
let userFavorites = new Set();

// Load user's favorites on page load
document.addEventListener('DOMContentLoaded', function() {
    loadUserFavorites();
    if (typeof loadFavoriteStatuses === 'function') {
        loadFavoriteStatuses();
    }
});

function loadUserFavorites() {
    fetch('/api/get_favorites')
        .then(response => response.json())
        .then(data => {
            userFavorites = new Set(data.favorites || []);
            updateFavoriteButtons();
        })
        .catch(error => {
            console.error('Error loading favorites:', error);
        });
}

// Function to load favorite statuses
function loadFavoriteStatuses() {
    fetch('/api/get_favorites')
        .then(response => response.json())
        .then(data => {
            const favorites = data.favorites || [];
            // Update UI with favorite statuses
            favorites.forEach(ticker => {
                const favoriteBtn = document.querySelector(`button[onclick*="${ticker}"]`);
                if (favoriteBtn && favoriteBtn.classList) {
                    favoriteBtn.classList.add('favorited');
                }
            });
        })
        .catch(error => {
            console.error('Error loading favorites:', error);
        });
}

function toggleFavorite(ticker, assetType = 'stock') {
    const isFavorite = userFavorites.has(ticker);
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
                userFavorites.add(ticker);
            } else {
                userFavorites.delete(ticker);
            }
            updateFavoriteButtons();

            // Show success message
            const message = action === 'add' ? 
                `Added ${ticker} to favorites` : 
                `Removed ${ticker} from favorites`;
            showToast(message, 'success');
        } else {
            showToast('Error updating favorites: ' + (data.error || 'Unknown error'), 'error');
        }
    })
    .catch(error => {
        console.error('Error toggling favorite:', error);
        showToast('Network error updating favorites', 'error');
    });
}

function updateFavoriteButtons() {
    // Update all favorite buttons on the page
    document.querySelectorAll('[data-ticker]').forEach(button => {
        const ticker = button.getAttribute('data-ticker');
        const icon = button.querySelector('i[data-feather]');

        if (userFavorites.has(ticker)) {
            // Safely add class
            if (button.classList) {
                button.classList.add('favorited');
            } else {
                button.className += ' favorited';
            }
            if (icon) {
                icon.setAttribute('data-feather', 'heart');
            }
        } else {
            // Safely remove class
            if (button.classList) {
                button.classList.remove('favorited');
            } else {
                button.className = button.className.replace(/\bfavorited\b/g, '').trim();
            }
            if (icon) {
                icon.setAttribute('data-feather', 'heart');
            }
        }
    });

    // Refresh feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

function showToast(message, type = 'info') {
    // Simple toast implementation
    const toast = document.createElement('div');
    toast.className = `alert alert-${type === 'success' ? 'success' : 'danger'} position-fixed`;
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
    toast.innerHTML = `
        <div class="d-flex align-items-center">
            <i data-feather="${type === 'success' ? 'check-circle' : 'alert-circle'}" class="me-2"></i>
            ${message}
            <button type="button" class="btn-close ms-auto" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;

    document.body.appendChild(toast);

    // Replace feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }

    // Auto remove after 3 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 3000);
}