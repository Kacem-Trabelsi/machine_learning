const $ = (s, el = document) => el.querySelector(s);

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Points PCA du dernier rendu (pour redessiner au changement de thème). */
let lastPcaPoints = [];

function initThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "dark";
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("ml-theme", next);
    drawPca();
  });
}

/** En file://, les URLs "/api/..." sont invalides ; on pointe vers l’API locale (CORS déjà autorisé côté FastAPI). */
const API_BASE = (() => {
  if (typeof window.ML_API_BASE === "string" && window.ML_API_BASE) {
    return window.ML_API_BASE.replace(/\/$/, "");
  }
  if (location.protocol === "file:") {
    return "http://127.0.0.1:8765";
  }
  return "";
})();

function apiUrl(path) {
  if (!path.startsWith("/")) return path;
  return API_BASE ? `${API_BASE}${path}` : path;
}

async function api(path, opts = {}) {
  const r = await fetch(apiUrl(path), {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  const text = await r.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!r.ok) {
    const msg = typeof data === "object" && data?.detail ? JSON.stringify(data.detail) : text;
    throw new Error(msg || r.statusText);
  }
  return data;
}

function setHealth() {
  const el = $("#health");
  api("/api/health")
    .then((h) => {
      const med = h.medical_loaded ? "modèle médical chargé" : "modèle médical manquant";
      const hos = h.hospital_loaded ? "régression hôpital chargée" : "régression manquante";
      el.className = "health ok";
      el.innerHTML = `<span>API OK</span> — ${med} · ${hos}.`;
    })
    .catch((e) => {
      el.className = "health";
      el.textContent = `API indisponible : ${e.message}`;
    });
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const id = `panel-${btn.dataset.tab}`;
    const panel = document.getElementById(id);
    if (panel) panel.classList.add("active");
  });
});

function randBetween(min, max, decimals) {
  const u = min + Math.random() * (max - min);
  if (decimals === undefined) return Math.round(u);
  return Number(u.toFixed(decimals));
}

/**
 * @param {'negative_leaning' | 'positive_leaning'} profile — plages inspirées Medicaldataset ; le modèle peut exceptionnellement contredire l’intention.
 */
function randomMedicalProfile(profile) {
  const neg = profile === "negative_leaning";
  let systolic = randBetween(neg ? 98 : 125, neg ? 155 : 215, 0);
  let diastolic = randBetween(neg ? 52 : 58, neg ? 92 : 105, 0);
  if (diastolic >= systolic - 10) {
    diastolic = Math.max(40, systolic - randBetween(25, 55, 0));
  }
  return {
    age: randBetween(20, 88, 0),
    gender: Math.random() < 0.5 ? 0 : 1,
    heart_rate: randBetween(neg ? 52 : 78, neg ? 98 : 145, 0),
    systolic_bp: systolic,
    diastolic_bp: diastolic,
    blood_sugar: randBetween(neg ? 72 : 140, neg ? 220 : 450, 0),
    ck_mb: randBetween(neg ? 0.25 : 3.5, neg ? 6 : 280, 2),
    troponin: randBetween(neg ? 0.001 : 0.12, neg ? 0.12 : 8, neg ? 4 : 3),
  };
}

function fillMedicalForm(s) {
  const f = $("#form-medical");
  if (!f) return;
  for (const [k, v] of Object.entries(s)) {
    const inp = f.elements.namedItem(k);
    if (inp) inp.value = String(v);
  }
}

function readMedicalBody(form) {
  const fd = new FormData(form);
  return {
    age: Number(fd.get("age")),
    gender: Number(fd.get("gender")),
    heart_rate: Number(fd.get("heart_rate")),
    systolic_bp: Number(fd.get("systolic_bp")),
    diastolic_bp: Number(fd.get("diastolic_bp")),
    blood_sugar: Number(fd.get("blood_sugar")),
    ck_mb: Number(fd.get("ck_mb")),
    troponin: Number(fd.get("troponin")),
  };
}

let medicalRandomNextPositiveLeaning = false;

document.getElementById("btn-sample-alternate")?.addEventListener("click", () => {
  const profile = medicalRandomNextPositiveLeaning ? "positive_leaning" : "negative_leaning";
  fillMedicalForm(randomMedicalProfile(profile));
  medicalRandomNextPositiveLeaning = !medicalRandomNextPositiveLeaning;
});

$("#form-medical").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = readMedicalBody(ev.target);
  const out = $("#out-medical");
  out.textContent = "Calcul…";
  try {
    const res = await api("/api/medical/predict", { method: "POST", body: JSON.stringify(body) });
    out.textContent =
      "Valeurs envoyées :\n" + JSON.stringify(body, null, 2) + "\n\nRéponse :\n" + JSON.stringify(res, null, 2);
  } catch (e) {
    out.textContent = "Erreur : " + e.message;
  }
});

$("#btn-example").addEventListener("click", async () => {
  const ta = $("#hospital-features");
  ta.value = "Chargement…";
  try {
    const ex = await api("/api/hospital/random-features");
    ta.value = ex.example.join(", ");
  } catch (e) {
    ta.value = "";
    $("#out-hospital").textContent = "Erreur tirage aléatoire : " + e.message;
  }
});

$("#btn-hospital").addEventListener("click", async () => {
  const raw = $("#hospital-features").value.trim();
  const parts = raw.split(/[\s,;]+/).filter(Boolean).map(Number);
  const out = $("#out-hospital");
  out.textContent = "Calcul…";
  try {
    const res = await api("/api/hospital/predict", {
      method: "POST",
      body: JSON.stringify({ features: parts }),
    });
    out.textContent = JSON.stringify(res, null, 2);
  } catch (e) {
    out.textContent = "Erreur : " + e.message;
  }
});

function drawPca(newPoints) {
  if (newPoints && newPoints.length > 0) {
    lastPcaPoints = newPoints;
  }
  const pts = lastPcaPoints;
  const canvas = $("#pca-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = cssVar("--canvas-bg", "#0d1219");
  ctx.fillRect(0, 0, w, h);
  if (!pts.length) return;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const pad = 40;
  const sx = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (w - 2 * pad);
  const sy = (y) => h - pad - ((y - minY) / (maxY - minY || 1)) * (h - 2 * pad);
  ctx.fillStyle = cssVar("--pca-dot", "rgba(56, 189, 248, 0.42)");
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = cssVar("--pca-axis", "#94a3b8");
  ctx.font = '500 12px "JetBrains Mono", ui-monospace, sans-serif';
  ctx.fillText("PC₁", w / 2 - 10, h - 10);
  ctx.save();
  ctx.translate(14, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("PC₂", 0, 0);
  ctx.restore();
}

$("#btn-pca").addEventListener("click", async () => {
  try {
    const data = await api("/api/clustering/pca2d?limit=1200");
    drawPca(data.points);
  } catch (e) {
    console.error(e);
  }
});

if (location.protocol === "file:") {
  const hint = document.getElementById("file-protocol-hint");
  if (hint) hint.hidden = false;
}

initThemeToggle();
setHealth();
api("/api/clustering/pca2d?limit=1200")
  .then((data) => drawPca(data.points))
  .catch(() => {});
