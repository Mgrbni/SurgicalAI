// SurgicalAI Professional Dashboard - Service Worker
// Basic PWA functionality for offline capabilities

const CACHE_NAME = 'surgicalai-dashboard-v1.0.0';
const urlsToCache = [
  '/',
  '/index.html',
  '/styles.css',
  '/api.js',
  'https://cdn.tailwindcss.com',
  'https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js',
  'https://unpkg.com/lucide@latest/dist/umd/lucide.js'
];

// Install event - cache resources
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('SurgicalAI Dashboard cache opened');
        return cache.addAll(urlsToCache);
      })
      .catch((error) => {
        console.warn('Cache installation failed:', error);
      })
  );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Return cached version or fetch from network
        if (response) {
          return response;
        }
        
        // Clone the request for caching
        const fetchRequest = event.request.clone();
        
        return fetch(fetchRequest).then((response) => {
          // Check if valid response
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }
          
          // Clone response for caching
          const responseToCache = response.clone();
          
          caches.open(CACHE_NAME)
            .then((cache) => {
              cache.put(event.request, responseToCache);
            });
          
          return response;
        }).catch(() => {
          // Return offline page or error for API calls
          if (event.request.url.includes('/api/')) {
            return new Response(
              JSON.stringify({ error: { message: 'Offline mode - API unavailable' } }),
              { 
                status: 503,
                headers: { 'Content-Type': 'application/json' }
              }
            );
          }
          
          // For other requests, try to return cached content
          return caches.match('/index.html');
        });
      })
  );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Handle background sync for offline form submissions
self.addEventListener('sync', (event) => {
  if (event.tag === 'surgical-analysis') {
    event.waitUntil(syncAnalysisData());
  }
});

// Sync analysis data when online
async function syncAnalysisData() {
  try {
    // Retrieve offline submissions from IndexedDB
    const offlineSubmissions = await getOfflineSubmissions();
    
    for (const submission of offlineSubmissions) {
      try {
        await fetch('/api/analyze', {
          method: 'POST',
          body: submission.formData
        });
        
        // Remove successful submission
        await removeOfflineSubmission(submission.id);
      } catch (error) {
        console.warn('Failed to sync submission:', error);
      }
    }
  } catch (error) {
    console.warn('Background sync failed:', error);
  }
}

// IndexedDB helpers (simplified)
async function getOfflineSubmissions() {
  // Placeholder for IndexedDB implementation
  return [];
}

async function removeOfflineSubmission(id) {
  // Placeholder for IndexedDB implementation
  console.log('Removing offline submission:', id);
}

// Push notification handling
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'SurgicalAI analysis update',
    icon: '/icon-192x192.png',
    badge: '/icon-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Results',
        icon: '/icon-view.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/icon-close.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('SurgicalAI Dashboard', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});
