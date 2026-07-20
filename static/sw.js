/* ═══════════════════════════════════════════════════════════════════════
   DHUN AI — Service Worker
   Implements: Cache-first for app shell, network-first for API calls,
               stale-while-revalidate for face-api models.
═══════════════════════════════════════════════════════════════════════ */

const CACHE_NAME    = 'dhun-ai-v1';
const MODEL_CACHE   = 'dhun-models-v1';
const OFFLINE_PAGE  = '/';

// App shell — cache on install
const SHELL_ASSETS = [
  '/',
  '/static/style.css',
  '/static/script.js',
  '/static/manifest.json',
];

// face-api.js models — heavy, long-lived cache
const MODEL_URLS = [
  'https://vladmandic.github.io/face-api/model/tiny_face_detector_model-weights_manifest.json',
  'https://vladmandic.github.io/face-api/model/face_landmark_68_model-weights_manifest.json',
  'https://vladmandic.github.io/face-api/model/face_recognition_model-weights_manifest.json',
];

// ── Install ────────────────────────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(SHELL_ASSETS).catch(e => console.warn('[SW] Shell cache failed:', e)))
      .then(() => self.skipWaiting())
  );
});

// ── Activate ───────────────────────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k !== CACHE_NAME && k !== MODEL_CACHE)
          .map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// ── Fetch ──────────────────────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET and cross-origin (except CDN models)
  if (request.method !== 'GET') return;

  // API calls → Network first, no cache
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }

  // face-api.js models → Cache first (very long-lived)
  if (url.href.includes('face-api.js') || url.href.includes('vladmandic')) {
    event.respondWith(cacheFirst(request, MODEL_CACHE));
    return;
  }

  // Chart.js / Google Fonts → Cache first
  if (url.href.includes('cdn.jsdelivr.net') || url.href.includes('fonts.googleapis')) {
    event.respondWith(cacheFirst(request, CACHE_NAME));
    return;
  }

  // App shell & static → Cache first, network fallback
  event.respondWith(cacheFirst(request, CACHE_NAME));
});

// ── Strategies ────────────────────────────────────────────────────────────

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    return response;
  } catch {
    // Offline — return cached index for navigation
    if (request.mode === 'navigate') {
      const cached = await caches.match(OFFLINE_PAGE);
      return cached || new Response('Offline — please reconnect.', { status: 503 });
    }
    return new Response('Offline', { status: 503 });
  }
}

async function cacheFirst(request, cacheName = CACHE_NAME) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok && response.status < 400) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    if (request.mode === 'navigate') {
      return caches.match(OFFLINE_PAGE) || new Response('Offline', { status: 503 });
    }
    return new Response('Offline', { status: 503 });
  }
}

// ── Background Sync (future: queue failed song saves) ─────────────────────
self.addEventListener('sync', event => {
  if (event.tag === 'sync-songs') {
    console.log('[SW] Background sync triggered');
  }
});

// ── Push Notifications (placeholder) ─────────────────────────────────────
self.addEventListener('push', event => {
  if (!event.data) return;
  const data = event.data.json();
  self.registration.showNotification(data.title || 'DHUN AI', {
    body: data.body || 'Your song is ready!',
    icon: '/static/icon-192.png',
    badge: '/static/icon-192.png',
    tag: 'dhun-notification',
  });
});
