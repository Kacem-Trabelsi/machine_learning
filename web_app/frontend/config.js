/**
 * Base URL de l’API (sans slash final).
 * - "" : même origine que la page → ex. http://127.0.0.1:8000 avec uvicorn (recommandé en local).
 * - "https://..." : backend hébergé ailleurs (Render, etc.) — doit exposer les mêmes routes (/api/medical/recommend inclus).
 */
window.ML_API_BASE = "";

/* Décommenter si le frontend statique est servi ailleurs que l’API et après déploiement de la dernière version du backend :
window.ML_API_BASE = "https://machine-learning-api-04an.onrender.com";
*/
