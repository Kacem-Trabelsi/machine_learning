/**
 * Copie le front statique vers dist/ pour Vercel (répertoire de sortie à la racine).
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const src = path.join(root, "web_app", "frontend");
const dest = path.join(root, "dist");

if (!fs.existsSync(src)) {
  console.error("Dossier manquant:", src);
  process.exit(1);
}

fs.rmSync(dest, { recursive: true, force: true });
fs.mkdirSync(dest, { recursive: true });
fs.cpSync(src, dest, { recursive: true });
console.log("Vercel: copié web_app/frontend -> dist");
