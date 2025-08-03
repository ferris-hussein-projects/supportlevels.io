const CACHE_NAME = 'stock-tracker-v1';
const urlsToCache = [
  '/',
  '/static/css/style.css',
  '/static/js/charts.js',
  'https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css',
  'https://unpkg.com/feather-icons',
  'https://cdn.jsdelivr.net/npm/chart.js'
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      }
    )
  );
});