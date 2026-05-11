/**
 * Base URL de l’API (sans slash final), définie automatiquement :
 * - localhost / 127.0.0.1 : "" → même machine que Uvicorn (ex. :8000).
 * - Déploiement (Vercel, GitHub Pages, etc.) : URL Render ci-dessous, sinon les requêtes
 *   /api/* partent vers l’hébergeur statique et renvoient 404 (NOT_FOUND).
 *
 * Surcharge manuelle : définir window.ML_API_BASE avant ce script, ou après :
 *   localStorage.setItem("ml-api-base-override", "https://votre-api.example.com");
 */
(function () {
  var OVERRIDE_KEY = "ml-api-base-override";
  var stored = localStorage.getItem(OVERRIDE_KEY);
  if (stored === "local" || stored === "same-origin") {
    window.ML_API_BASE = "";
    return;
  }
  if (stored && /^https?:\/\//.test(stored)) {
    window.ML_API_BASE = stored.replace(/\/$/, "");
    return;
  }

  var host = (window.location.hostname || "").toLowerCase();
  if (host === "localhost" || host === "127.0.0.1") {
    window.ML_API_BASE = "";
    return;
  }

  /* Backend public (Render) — mettre à jour si l’URL du service change */
  window.ML_API_BASE = "https://machine-learning-api-04an.onrender.com";
})();
