/**
 * MEDIRAG-XAI — Centralized Frontend Configuration
 * 
 * Sets the backend API base URL for both local development and production deployments.
 * 
 * -----------------------------------------------------------------------------------
 * HOW TO CONFIGURE IN PRODUCTION:
 * 
 * 1. UNIFIED DEPLOYMENT (Frontend & Backend on the same server, e.g. Render/Docker):
 *    Leave API_BASE_URL as "" (empty string). All requests will use relative paths.
 * 
 * 2. DECOUPLED DEPLOYMENT (Frontend on Vercel/Netlify/GitHub Pages & Backend on Render):
 *    Option A: Edit API_BASE_URL below and set your live backend URL, e.g.:
 *              API_BASE_URL: 'https://your-medirag-backend.onrender.com'
 *    Option B: In browser console, run:
 *              localStorage.setItem('MEDIRAG_API_URL', 'https://your-medirag-backend.onrender.com');
 * -----------------------------------------------------------------------------------
 */

const MEDIRAG_CONFIG = {
  // Set your production backend URL here if hosted separately, or leave "" for auto/same-origin
  API_BASE_URL: (function() {
    if (typeof window !== 'undefined') {
      // 1. Check for runtime window variable or localStorage override
      if (window.MEDIRAG_API_URL) return window.MEDIRAG_API_URL;
      const stored = localStorage.getItem('MEDIRAG_API_URL');
      if (stored) return stored;

      // 2. Local development auto-detection
      const isLocalhost = Boolean(
        window.location.hostname === 'localhost' ||
        window.location.hostname === '127.0.0.1' ||
        window.location.hostname === '[::1]'
      );

      if (isLocalhost) {
        // If served from FastAPI (port 8000), relative path works directly
        // If served from Live Server or static dev server (port 5500, 3000, etc.), point to backend port 8000
        return window.location.port === '8000' ? '' : 'http://localhost:8000';
      }

      // 3. Deployed default: relative path (same origin)
      // When deploying frontend on Vercel/Netlify, replace '' with your backend URL or set via localStorage
      return '';
    }
    return '';
  })(),
};

// Export to global window object
if (typeof window !== 'undefined') {
  window.APP_CONFIG = MEDIRAG_CONFIG;
}
