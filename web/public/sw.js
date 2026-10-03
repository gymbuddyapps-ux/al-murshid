// Service worker: makes the app work without internet after the first visit.
//
// When the app is installed, every file of the app (pages, code, fonts, models)
// is saved in the browser's cache. Later, files are answered from that cache.
// Nothing is ever sent anywhere: this file only reads the app's own files.

// The two values below are filled in automatically when the app is built
// (see vite.config.js).
const FILES = "__FILES__";
const CACHE = 'jisr-__VERSION__';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(FILES))
      .then(() => self.skipWaiting()),
  );
});

// Delete caches of older versions of the app.
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) => Promise.all(names.filter((n) => n !== CACHE).map((n) => caches.delete(n))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;

  // Opening the app (a page navigation) always gets the main page.
  // ignoreVary: some servers add a "Vary" header that would otherwise stop
  // module scripts from matching their cached copy.
  const key = event.request.mode === 'navigate' ? './index.html' : event.request;
  event.respondWith(
    caches
      .match(key, { ignoreSearch: true, ignoreVary: true })
      .then((cached) => cached || fetch(event.request)),
  );
});
