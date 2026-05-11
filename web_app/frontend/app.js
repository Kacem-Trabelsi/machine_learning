const $ = (s, el = document) => el.querySelector(s);

let wizardStep = 0;
const WIZARD_LAST = 2;

const HOSPITAL_GROUP_LABELS = [
  "Patient & posologie",
  "Voie d’administration & fréquence",
  "Diagnostics",
  "Antibiothérapie (schéma one-hot)",
  "Indications cliniques",
];

function hospitalGroupIndex(featureName) {
  const n = featureName;
  if (n === "Log_Dosage" || n === "Scaled_Age" || n === "Gender_Male") return 0;
  if (n.startsWith("Route_") || n.startsWith("Frequency_")) return 1;
  if (n.startsWith("Diagnosis_")) return 2;
  if (n.startsWith("Name of Drug_")) return 3;
  if (n.startsWith("Indication_")) return 4;
  return 0;
}

/** Groupes non vides après chargement des noms côté API. */
let hospitalVisibleGroups = [];
let hospitalSubIdx = 0;

function initThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "dark";
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("ml-theme", next);
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

const MEDICAL_FIELD_LABELS = {
  age: "Âge",
  gender: "Genre (0 / 1)",
  heart_rate: "Fréquence cardiaque",
  systolic_bp: "Pression systolique",
  diastolic_bp: "Pression diastolique",
  blood_sugar: "Glycémie",
  ck_mb: "CK-MB",
  troponin: "Troponine",
};

function formatFrPct(x, digits = 2) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return `${(x * 100).toLocaleString("fr-FR", { maximumFractionDigits: digits, minimumFractionDigits: digits })} %`;
}

function formatFrNum(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return Number.isInteger(x) ? String(x) : String(x.toLocaleString("fr-FR", { maximumFractionDigits: 4 }));
}

function clearResultStack(el) {
  if (!el) return;
  el.hidden = true;
  el.replaceChildren();
}

function renderResultLoading(el, msg = "Calcul en cours…") {
  if (!el) return;
  el.hidden = false;
  el.replaceChildren();
  const p = document.createElement("p");
  p.className = "result-loading";
  p.textContent = msg;
  el.appendChild(p);
}

function renderResultError(el, message) {
  if (!el) return;
  el.hidden = false;
  el.replaceChildren();
  const box = document.createElement("div");
  box.className = "result-card result-card--error";
  box.textContent = message;
  el.appendChild(box);
}

function renderMedicalResult(el, body, res) {
  if (!el) return;
  el.hidden = false;
  el.replaceChildren();

  const pos = res.prediction_label === "positive";
  const pp = typeof res.probability_positive === "number" ? Math.min(1, Math.max(0, res.probability_positive)) : 0;
  const pn = 1 - pp;

  const hero = document.createElement("div");
  hero.className = "result-hero " + (pos ? "result-hero--risk" : "result-hero--ok");
  const verdict = document.createElement("div");
  verdict.className = "result-hero__verdict";
  const badge = document.createElement("span");
  badge.className = "result-badge " + (pos ? "result-badge--alert" : "result-badge--ok");
  badge.textContent = pos ? "Positif — risque associé" : "Négatif — risque modéré";
  verdict.appendChild(badge);
  const sub = document.createElement("p");
  sub.className = "result-hero__sub";
  sub.textContent = pos
    ? "Le modèle estime une probabilité élevée de classe « positive ». À interpréter avec prudence (démonstration ML)."
    : "Le modèle estime plutôt une classe « négative ». Valeur indicative, non diagnostique.";
  hero.appendChild(verdict);
  hero.appendChild(sub);

  const proba = document.createElement("div");
  proba.className = "result-proba";
  const meta = document.createElement("div");
  meta.className = "result-proba__meta";
  meta.textContent = `Probabilité positive : ${formatFrPct(pp)} · négative : ${formatFrPct(pn)}`;
  const track = document.createElement("div");
  track.className = "result-proba__track";
  const fill = document.createElement("div");
  fill.className = "result-proba__fill";
  fill.style.width = `${pp * 100}%`;
  track.appendChild(fill);
  proba.appendChild(meta);
  proba.appendChild(track);

  const card = document.createElement("div");
  card.className = "result-card";
  const h3 = document.createElement("h3");
  h3.className = "result-card__title";
  h3.textContent = "Signes saisis";
  const grid = document.createElement("dl");
  grid.className = "result-kv";
  for (const [k, label] of Object.entries(MEDICAL_FIELD_LABELS)) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = formatFrNum(body[k]);
    grid.appendChild(dt);
    grid.appendChild(dd);
  }
  card.appendChild(h3);
  card.appendChild(grid);

  const details = document.createElement("details");
  details.className = "result-details";
  const sum = document.createElement("summary");
  sum.textContent = "Données brutes (JSON)";
  const pre = document.createElement("pre");
  pre.className = "result-details__pre";
  pre.textContent = JSON.stringify({ input: body, output: res }, null, 2);
  details.appendChild(sum);
  details.appendChild(pre);

  el.appendChild(hero);
  el.appendChild(proba);
  el.appendChild(card);
  el.appendChild(details);
}

function renderHospitalResult(el, res) {
  if (!el) return;
  el.hidden = false;
  el.replaceChildren();
  const days = res.predicted_duration_days;

  const hero = document.createElement("div");
  hero.className = "result-hero result-hero--neutral";
  const cap = document.createElement("p");
  cap.className = "result-hero__sub result-hero__sub--tight";
  cap.textContent = "Durée d’hospitalisation estimée par le modèle";
  const row = document.createElement("div");
  row.className = "result-hospital-row";
  const val = document.createElement("span");
  val.className = "result-hospital-value";
  val.textContent =
    typeof days === "number" && !Number.isNaN(days)
      ? days.toLocaleString("fr-FR", { maximumFractionDigits: 2 })
      : "—";
  const unit = document.createElement("span");
  unit.className = "result-hospital-unit";
  unit.textContent = " jours";
  row.appendChild(val);
  row.appendChild(unit);
  hero.appendChild(cap);
  hero.appendChild(row);

  const details = document.createElement("details");
  details.className = "result-details";
  const sum = document.createElement("summary");
  sum.textContent = "Réponse API (JSON)";
  const pre = document.createElement("pre");
  pre.className = "result-details__pre";
  pre.textContent = JSON.stringify(res, null, 2);
  details.appendChild(sum);
  details.appendChild(pre);

  el.appendChild(hero);
  el.appendChild(details);
}

function fillRecommendForm(s) {
  const f = document.getElementById("form-recommend");
  if (!f) return;
  for (const [k, v] of Object.entries(s)) {
    const inp = f.elements.namedItem(k);
    if (inp) inp.value = String(v);
  }
}

function syncRecommendFormFromMedical() {
  const src = document.getElementById("form-medical");
  const dst = document.getElementById("form-recommend");
  if (!src || !dst) return;
  for (const k of Object.keys(MEDICAL_FIELD_LABELS)) {
    const si = src.elements.namedItem(k);
    const di = dst.elements.namedItem(k);
    if (si && di) di.value = si.value;
  }
}

function renderRecommendationResult(el, body, res) {
  if (!el) return;
  el.hidden = false;
  el.replaceChildren();
  const frac =
    typeof res.neighbor_fraction_positive === "number" ? res.neighbor_fraction_positive : 0;
  const maj = res.neighbor_majority_label || "—";
  const pos = maj === "positive";

  const hero = document.createElement("div");
  hero.className = "result-hero result-hero--neutral";
  const verdict = document.createElement("div");
  verdict.className = "result-hero__verdict";
  const sp = document.createElement("span");
  sp.className = "result-badge " + (pos ? "result-badge--alert" : "result-badge--ok");
  sp.textContent = `Consensus des k=${res.k_neighbors} voisins : ${maj}`;
  verdict.appendChild(sp);
  const sub = document.createElement("p");
  sub.className = "result-hero__sub";
  sub.textContent = `Part de cas « positive » parmi ces voisins (entraînement) : ${formatFrPct(frac)}. Distance : ${res.metric || "euclidean"} dans l’espace prétraité.`;
  hero.appendChild(verdict);
  hero.appendChild(sub);
  el.appendChild(hero);

  const clf = res.meta && res.meta.classifier_hint;
  if (clf) {
    const cmp = document.createElement("div");
    cmp.className = "result-card";
    const t = document.createElement("h3");
    t.className = "result-card__title";
    t.textContent = "Classifieur principal (même saisie)";
    const p = document.createElement("p");
    p.className = "result-hero__sub";
    p.style.margin = "0";
    p.textContent = `${clf.prediction_label} · P(positif) = ${formatFrPct(clf.probability_positive)} — lecture complémentaire au vote des voisins.`;
    cmp.appendChild(t);
    cmp.appendChild(p);
    el.appendChild(cmp);
  }

  const tableCard = document.createElement("div");
  tableCard.className = "result-card";
  const h3 = document.createElement("h3");
  h3.className = "result-card__title";
  h3.textContent = "Voisins les plus proches (historique train)";
  const tbl = document.createElement("table");
  tbl.className = "neighbor-table";
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  for (const text of ["Rang", "Distance", "Issue"]) {
    const th = document.createElement("th");
    th.textContent = text;
    trh.appendChild(th);
  }
  thead.appendChild(trh);
  tbl.appendChild(thead);
  const tb = document.createElement("tbody");
  for (const n of res.neighbors || []) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    td1.textContent = String(n.rank);
    const td2 = document.createElement("td");
    td2.textContent = typeof n.distance === "number" ? n.distance.toFixed(4) : "—";
    const td3 = document.createElement("td");
    td3.textContent = String(n.historical_outcome);
    tr.appendChild(td1);
    tr.appendChild(td2);
    tr.appendChild(td3);
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  tableCard.appendChild(h3);
  tableCard.appendChild(tbl);
  el.appendChild(tableCard);

  const hv = res.meta && res.meta.sidecar && res.meta.sidecar.holdout_neighbor_majority;
  if (hv) {
    const ev = document.createElement("div");
    ev.className = "result-card";
    const th2 = document.createElement("h3");
    th2.className = "result-card__title";
    th2.textContent = "Référence notebook (hold-out, vote k-NN)";
    const pp = document.createElement("p");
    pp.className = "result-hero__sub";
    pp.style.margin = "0";
    pp.textContent = `Accuracy ${(hv.accuracy * 100).toFixed(1)} % · Rappel positif ${(hv.recall_positive * 100).toFixed(1)} % · F1 ${hv.f1.toFixed(3)}`;
    ev.appendChild(th2);
    ev.appendChild(pp);
    el.appendChild(ev);
  }

  const details = document.createElement("details");
  details.className = "result-details";
  const sum = document.createElement("summary");
  sum.textContent = "Réponse API (JSON)";
  const pre = document.createElement("pre");
  pre.className = "result-details__pre";
  pre.textContent = JSON.stringify(res, null, 2);
  details.appendChild(sum);
  details.appendChild(pre);
  el.appendChild(details);
}

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
  if (!el) return;
  api("/api/health")
    .then((h) => {
      const med = h.medical_loaded ? "modèle médical chargé" : "modèle médical manquant";
      const hos = h.hospital_loaded ? "régression hôpital chargée" : "régression manquante";
      const rec = h.recommendation_loaded ? "recommandation k-NN chargée" : "recommandation manquante";
      el.className = "health ok";
      el.innerHTML = `<span>API OK</span> — ${med} · ${hos} · ${rec}.`;
    })
    .catch((e) => {
      el.className = "health";
      el.textContent =
        `API indisponible : ${e.message} — en local : ouvrir via localhost et le backend (ex. :8000), ou localStorage.setItem("ml-api-base-override","https://…onrender.com"). Sur le déploiement statique, config.js pointe déjà l’API Render ; un NOT_FOUND indique souvent un site sans cette route (redéployer l’API) ou cache (Ctrl+F5).`;
    });
}

function setWizardStep(n) {
  wizardStep = Math.max(0, Math.min(WIZARD_LAST, n));
  const panes = document.querySelectorAll(".wizard-pane");
  const items = document.querySelectorAll("#wizard-progress .wizard-progress__item");

  panes.forEach((p) => {
    const i = Number(p.dataset.wizardPane);
    const active = i === wizardStep;
    p.classList.toggle("is-active", active);
    if (active) {
      p.removeAttribute("hidden");
      p.setAttribute("aria-hidden", "false");
    } else {
      p.setAttribute("hidden", "");
      p.setAttribute("aria-hidden", "true");
    }
  });

  items.forEach((el, i) => {
    el.classList.toggle("is-active", i === wizardStep);
    /** @type {HTMLElement} */
    const idx = el.querySelector(".wizard-progress__idx");
    if (idx) idx.textContent = String(i + 1);
    if (i === wizardStep) el.setAttribute("aria-current", "step");
    else el.removeAttribute("aria-current");
    const done = i < wizardStep;
    el.classList.toggle("is-done", done);
  });

  const prev = document.getElementById("btn-wizard-prev");
  const next = document.getElementById("btn-wizard-next");
  const hint = document.getElementById("wizard-footer-hint");
  if (prev) prev.disabled = wizardStep <= 0;
  if (next) {
    if (wizardStep >= WIZARD_LAST) {
      next.textContent = "Fin du parcours";
      next.disabled = true;
    } else {
      next.textContent = "Étape suivante →";
      next.disabled = false;
    }
  }
  if (hint) {
    const hints = [
      "Étape 1 / 3 — classification cardiaque (positif / négatif).",
      "Étape 2 / 3 — durée d’hospitalisation (régression).",
      "Étape 3 / 3 — cas similaires dans le jeu d’entraînement (k-NN).",
    ];
    hint.textContent = hints[wizardStep] ?? "";
  }

  if (wizardStep === WIZARD_LAST) syncRecommendFormFromMedical();
}

function initWizardChrome() {
  document.querySelectorAll("#wizard-progress [data-wizard-goto]").forEach((btn) => {
    btn.addEventListener("click", () => setWizardStep(Number(btn.dataset.wizardGoto)));
  });
  document.getElementById("btn-wizard-prev")?.addEventListener("click", () => setWizardStep(wizardStep - 1));
  document.getElementById("btn-wizard-next")?.addEventListener("click", () => setWizardStep(wizardStep + 1));
}

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

$("#form-medical")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = readMedicalBody(ev.target);
  const out = $("#out-medical");
  if (!out) return;
  renderResultLoading(out);
  try {
    const res = await api("/api/medical/predict", { method: "POST", body: JSON.stringify(body) });
    renderMedicalResult(out, body, res);
  } catch (e) {
    renderResultError(out, "Erreur : " + e.message);
  }
});

let recommendRandomNextPositiveLeaning = false;

document.getElementById("btn-recommend-sync")?.addEventListener("click", () => {
  syncRecommendFormFromMedical();
});

document.getElementById("btn-recommend-random")?.addEventListener("click", () => {
  const profile = recommendRandomNextPositiveLeaning ? "positive_leaning" : "negative_leaning";
  fillRecommendForm(randomMedicalProfile(profile));
  recommendRandomNextPositiveLeaning = !recommendRandomNextPositiveLeaning;
});

$("#form-recommend")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = readMedicalBody(ev.target);
  const out = $("#out-recommend");
  if (!out) return;
  renderResultLoading(out);
  try {
    const res = await api("/api/medical/recommend", { method: "POST", body: JSON.stringify(body) });
    renderRecommendationResult(out, body, res);
  } catch (e) {
    renderResultError(out, "Erreur : " + e.message);
  }
});

/** Ordre des noms de caractéristiques hôpital (aligné sur l’API). */
let hospitalFeatureNames = [];

function hospitalForm() {
  return document.getElementById("form-hospital");
}

function readHospitalFeaturesFromForm(form) {
  return hospitalFeatureNames.map((_, i) => {
    const el = form.elements.namedItem(`hf_${i}`);
    if (!el || el.value === "") return NaN;
    return Number(el.value);
  });
}

function setHospitalFieldValues(values) {
  const form = hospitalForm();
  if (!form || values.length !== hospitalFeatureNames.length) return;
  values.forEach((v, i) => {
    const el = form.elements.namedItem(`hf_${i}`);
    if (el) el.value = String(v);
  });
}

function showHospitalSubStep(idx) {
  if (!hospitalVisibleGroups.length) return;
  hospitalSubIdx = Math.max(0, Math.min(hospitalVisibleGroups.length - 1, idx));
  const wrap = document.getElementById("hospital-fields-wrap");
  wrap?.querySelectorAll(".hospital-sub-pane").forEach((pane, i) => {
    if (i === hospitalSubIdx) pane.removeAttribute("hidden");
    else pane.setAttribute("hidden", "");
  });
  document.querySelectorAll("#hospital-step-pills .step-pill").forEach((pill, i) => {
    pill.classList.toggle("is-active", i === hospitalSubIdx);
    pill.setAttribute("aria-selected", i === hospitalSubIdx ? "true" : "false");
  });
  const g = hospitalVisibleGroups[hospitalSubIdx];
  const cap = document.getElementById("hospital-substep-caption");
  if (cap && g) {
    cap.textContent = `Sous-étape ${hospitalSubIdx + 1} / ${hospitalVisibleGroups.length} — ${HOSPITAL_GROUP_LABELS[g.gi]}`;
  }
  const prev = document.getElementById("btn-hospital-sub-prev");
  const next = document.getElementById("btn-hospital-sub-next");
  if (prev) prev.disabled = hospitalSubIdx <= 0;
  if (next) next.disabled = hospitalSubIdx >= hospitalVisibleGroups.length - 1;
}

let hospitalSubNavBound = false;

function bindHospitalSubNavOnce() {
  if (hospitalSubNavBound) return;
  hospitalSubNavBound = true;
  document.getElementById("btn-hospital-sub-prev")?.addEventListener("click", () => showHospitalSubStep(hospitalSubIdx - 1));
  document.getElementById("btn-hospital-sub-next")?.addEventListener("click", () => showHospitalSubStep(hospitalSubIdx + 1));
}

async function initHospitalForm() {
  const wrap = document.getElementById("hospital-fields-wrap");
  const status = document.getElementById("hospital-fields-status");
  const inner = document.getElementById("hospital-inner-wizard");
  const pills = document.getElementById("hospital-step-pills");
  if (!wrap || !pills) return;
  bindHospitalSubNavOnce();
  try {
    const data = await api("/api/hospital/feature-names");
    hospitalFeatureNames = data.feature_names || [];
    const buckets = [[], [], [], [], []];
    hospitalFeatureNames.forEach((name, i) => {
      buckets[hospitalGroupIndex(name)].push({ name, i });
    });
    hospitalVisibleGroups = buckets
      .map((items, gi) => ({ gi, items }))
      .filter((x) => x.items.length > 0);

    if (status) status.setAttribute("hidden", "");
    if (inner) inner.removeAttribute("hidden");

    wrap.innerHTML = "";
    pills.innerHTML = "";

    hospitalVisibleGroups.forEach((group, si) => {
      const pane = document.createElement("div");
      pane.className = "hospital-sub-pane";
      if (si !== 0) pane.setAttribute("hidden", "");
      const grid = document.createElement("div");
      grid.className = "form-grid form-grid--hospital";
      group.items.forEach(({ name, i }) => {
        const label = document.createElement("label");
        label.className = "hospital-field";
        const span = document.createElement("span");
        span.className = "hospital-field-name";
        span.textContent = name;
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.required = true;
        input.name = `hf_${i}`;
        input.id = `hf_${i}`;
        input.autocomplete = "off";
        label.appendChild(span);
        label.appendChild(input);
        grid.appendChild(label);
      });
      pane.appendChild(grid);
      wrap.appendChild(pane);

      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "step-pill";
      if (si === 0) pill.classList.add("is-active");
      pill.textContent = String(si + 1);
      pill.title = HOSPITAL_GROUP_LABELS[group.gi];
      pill.setAttribute("role", "tab");
      pill.setAttribute("aria-selected", si === 0 ? "true" : "false");
      pill.addEventListener("click", () => showHospitalSubStep(si));
      pills.appendChild(pill);
    });

    hospitalSubIdx = 0;
    showHospitalSubStep(0);
  } catch (e) {
    hospitalFeatureNames = [];
    hospitalVisibleGroups = [];
    if (inner) inner.setAttribute("hidden", "");
    if (status) {
      status.removeAttribute("hidden");
      status.textContent =
        "Impossible de charger les champs (API ou modèle hôpital). Détail : " +
        e.message +
        " — ouvrez le site via le même hôte que l’API (ex. 127.0.0.1:8000) et ML_API_BASE vide dans config.js.";
      status.className = "hospital-fields-status hospital-fields-status--error";
    }
  }
}

document.getElementById("btn-example")?.addEventListener("click", async () => {
  const out = $("#out-hospital");
  if (!hospitalFeatureNames.length) {
    if (out) renderResultError(out, "Formulaire hôpital non chargé — vérifiez l’API et le modèle.");
    return;
  }
  if (out) clearResultStack(out);
  try {
    const ex = await api("/api/hospital/random-features");
    setHospitalFieldValues(ex.example);
  } catch (e) {
    if (out) renderResultError(out, "Erreur tirage aléatoire : " + e.message);
  }
});

hospitalForm()?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const parts = readHospitalFeaturesFromForm(form);
  const out = $("#out-hospital");
  if (!out) return;
  if (parts.some((n) => Number.isNaN(n))) {
    renderResultError(out, "Complétez tous les champs (parcourir chaque sous-étape).");
    return;
  }
  if (parts.length !== hospitalFeatureNames.length) {
    renderResultError(out, "Nombre de caractéristiques incorrect ; vérifiez le modèle côté serveur.");
    return;
  }
  renderResultLoading(out);
  try {
    const res = await api("/api/hospital/predict", {
      method: "POST",
      body: JSON.stringify({ features: parts }),
    });
    renderHospitalResult(out, res);
  } catch (e) {
    renderResultError(out, "Erreur : " + e.message);
  }
});

if (location.protocol === "file:") {
  const hint = document.getElementById("file-protocol-hint");
  if (hint) hint.hidden = false;
}

initThemeToggle();
initWizardChrome();
setWizardStep(0);
setHealth();
initHospitalForm();
