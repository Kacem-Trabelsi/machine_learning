# CRISP-DM End-to-End Report — Medical Classification Project

**Language:** English  
**Scope:** Binary classification on the Medical dataset (`Result`: negative / positive).  
**Artifacts (main paths):**  
`classification_Medical_data _set/01_Classification_Medical.ipynb` (Data Understanding),  
`classification_Medical_data _set/01b_DataPreparation_Medical.ipynb` (Data Preparation),  
`classification_Medical_data _set/01c_Modeling_Medical.ipynb` (Modeling),  
`classification_Medical_data _set/data/processed/medical/` (processed CSVs + `metadata.json`),  
`classification_Medical_data _set/models/` (saved model + selection JSON).

---

## How to read this document

- **Phases** follow the CRISP-DM sequence: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation (and notes on Deployment).
- **“What we did”** matches the notebooks and exports in this repository.
- **“What the 5 supervisors said”** summarizes independent methodological review of your latest results (stratified CV, hold-out test, high ensemble performance).

---

## Phase 1 — Business Understanding

### Intent

- **Business goal (high level):** Support clinical decision-making around severe cardiac risk by using patient-level tabular data (vitals and cardiac biomarkers).
- **Data science goal:** Learn a supervised **binary classifier** that predicts the `Result` label from features available in the Medical dataset.
- **Success criterion (clinical framing):** Prioritize **sensitivity (Recall)** for the **positive** class (`Result == positive` → encoded as `1`), because false negatives (missing a high-risk case) are often treated as more harmful than false positives (extra work-up) in a screening-style narrative.

### Note on documentation alignment

- The presentation / slides may use synonyms such as “heart attack” or `Heart_Attack` for exposition. In the **actual pipeline**, the target column is **`Result`** with explicit mapping **`negative → 0`**, **`positive → 1`** (see `metadata.json`). All modeling metrics use **positive class = 1**.

---

## Phase 2 — Data Understanding

### Notebook

`classification_Medical_data _set/01_Classification_Medical.ipynb`

### What we did

1. **Load raw data** from `Medicaldataset.csv` (in the same folder as the classification notebooks or resolved relative to project root).
2. **Copy governance (good practice):**
   - **`df_raw`:** immutable snapshot after load.
   - **`df_du` / `df`:** single working copy for exploration so we do not create uncontrolled duplicates.
3. **Schema and descriptive statistics:** `info()`, `describe()`, distributions.
4. **Univariate analysis:** histograms / KDE for numeric features.
5. **Correlation analysis:** Pearson heatmap + auxiliary diagnostics for highly correlated pairs (multicollinearity awareness).
6. **Data quality:**
   - Missing values, duplicate rows.
   - **Plausible range checks** for vitals (soft clinical bounds used as **data-quality screens**, not as automatic patient exclusion).
   - **IQR-based outlier rates** (flagging, not blind deletion — extremes may be clinically real).
   - **Skewness** notes for possible transforms in preparation.
7. **Bivariate analysis:** target (`Result`) vs features (e.g., boxplots by class).
8. **Technical audit section:** duplicate column names, QA summary table, and an **action log** (what we check vs what we defer to Data Preparation).

### Key findings (typical)

- No missing values in the raw table for this dataset version.
- A **small number** of heart-rate and systolic BP values fall outside soft plausible ranges — flagged for explicit handling in Data Preparation (not silently kept as raw errors if they are likely artifacts).
- Class imbalance is **moderate** (roughly **~61% positive / ~39% negative** in the stratified split used later — exact rates appear in notebook outputs).

---

## Phase 3 — Data Preparation

### Notebook

`classification_Medical_data _set/01b_DataPreparation_Medical.ipynb`

### Principles (industry-aligned for a course / prototype)

1. **Leakage prevention:** **Train/test split first** (or equivalently: any statistic used to transform data — imputation, scaling — must be **fit on training data only**).
2. **Reproducibility:** fixed **`random_state`**, exported **`metadata.json`** describing columns, bounds, and QA.

### What we did (pipeline summary)

1. **Robust path resolution** so the notebook runs from project root or from `classification_Medical_data _set/`.
2. **Immutable vs working copy:** `df_raw`, `df_prep`, then `df_work` for transformations.
3. **Schema contract:** required columns enforced.
4. **Clinical rule layer (data quality):**
   - Binary flags **`flag_hr_out_of_range`**, **`flag_sbp_out_of_range`** for values outside soft HR / systolic bounds.
   - Flagged raw vitals set to **`NaN`** so they are handled by **train-only median imputation** (not “hidden” inside the raw feature).
5. **Feature engineering:**
   - **`Pulse_Pressure`** = systolic − diastolic blood pressure.
   - **`CK_MB_Troponin_Ratio`** = CK-MB / max(Troponin, ε) to avoid divide-by-zero.
6. **Target encoding:** explicit map `negative → 0`, `positive → 1` (no ambiguous `LabelEncoder` ordering).
7. **Split:** **80% / 20%**, **`stratify=y`**, **`random_state=42`**.
8. **Train-only preprocessing:**
   - **`SimpleImputer(strategy='median')`** fit on `X_train` continuous columns, transform train & test.
   - **`RobustScaler`** fit on **imputed** continuous columns, transform train & test.
   - **Flags are not scaled** (binary indicators passed through).
9. **QA before export:** confirm no `NaN` / `inf` in processed matrices, record flag counts, optional complete-case sensitivity counts.
10. **Export:**
    - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
    - `metadata.json` (feature order, target mapping, bounds, QA summary).

### Why scaled features look “strange” (negative ages, odd gender values)

- **`RobustScaler`** centers and scales using **median** and **IQR** from **training data**.
- Values **below the training median** become **negative** in scaled space. That does **not** mean a negative age in years — it means “below median age” in standardized units.
- **`Gender`** may appear as values like **`0`** and **`-1`** after scaling if it was included in the scaled block: it remains a **two-level** categorical signal, just expressed in scaled coordinates. (Optional refinement: pass `Gender` through without scaling using `ColumnTransformer` — a documented improvement, not a correctness bug for distance-based models.)

### Outputs (current contract)

- **Train:** 1055 rows × **12** features.  
- **Test:** 264 rows × **12** features.  
- **Features (12):** base numerics + engineered `Pulse_Pressure`, `CK_MB_Troponin_Ratio` + rule flags.

---

## Phase 4 — Modeling

### Notebook

`classification_Medical_data _set/01c_Modeling_Medical.ipynb`

### What “Stratified cross-validation (train only) + hold-out test” means

**Plain language**

- You first reserve a **hold-out test set** that must not influence training decisions.
- On the **training portion only**, you run **stratified k-fold cross-validation**: the data are split into *k* folds, and **each fold keeps roughly the same proportion of classes** as the full training set. This matters when classes are not 50/50.
- You **average validation performance** across folds to compare models **without peeking at the test set**.
- After choosing the model type by CV, you **refit on the full training set** and report **one** evaluation on the hold-out test.

**Why it is standard**

- Using the **test set** to pick the “winner” makes the test set a validation set and **inflates** reported performance.
- **Stratification** avoids folds with zero (or nearly zero) positives.

**Procedure used in this project (numbered)**

1. Load processed `X_train`, `X_test`, `y_train`, `y_test` + `metadata.json`.
2. **StratifiedKFold (`k=5`, shuffled)** on **`X_train`, `y_train` only**.
3. For each candidate model, compute CV scores for **Recall**, **ROC-AUC**, **PR-AUC (average precision)**.
4. Record **`cv_winner`** = non-dummy model with highest **mean CV Recall**; apply **deploy policy** (default **Gradient Boosting** when in the zoo — override in notebook to export CV winner only).
5. Refit **every** candidate on the **full** training set (for fair reporting tables and confusion matrices).
6. Measure **hold-out test metrics once** per model (for transparency — the deploy model is fixed by step 4, not by sorting the test table).
7. **Learning curve** (train only, PR-AUC) and **calibration + Brier** on hold-out for the deploy model.

### Model zoo (candidates)

- **Dummy (stratified)** — sanity baseline.
- **Logistic Regression** — linear baseline (`class_weight='balanced'`).
- **Random Forest** — ensemble trees (`class_weight='balanced'`).
- **SVC (RBF)** — kernel method (expects scaled features — satisfied by Phase 3).
- **Gradient Boosting** — boosted trees.

**Positive class for metrics:** `1` = `positive`.

### Observed results (from your run — representative)

**Cross-validation (training only)**

- **Gradient Boosting** had the highest **mean CV Recall** (~0.986 in your screenshot), with low std — selected for export.
- **Random Forest** was extremely close and often **higher** on CV **PR-AUC / ROC-AUC** in your table — this is normal: different models optimize different trade-offs.

**Hold-out test**

- **Gradient Boosting** showed very strong test metrics (accuracy ~0.985, recall ~0.988, high ROC-AUC and PR-AUC in your output).
- **SVC** lagged on recall in your run (many false negatives in the confusion matrix) — consistent with harder decision boundaries / class separation behavior on this feature space.

### Artifacts saved

- **`medical_best_model.pkl`** — fitted **CV-selected** model (in your successful run: **Gradient Boosting**) refit on the **full training set**.
- **`medical_model_selection.json`** — documents `selected_model`, `cv_winner_by_mean_recall`, `deploy_model_choice`, CV recall, test metrics **including Brier**, seed, fold count, path to artifact.
- **`medical_rf_model.pkl`** — written **only if** the **selected** model is **Random Forest** (backward compatibility alias). If GB is selected, **do not** assume this file exists.

### Feature importance plot

- The notebook plots **Random Forest** impurity-based importances for **interpretability**.
- **Important for presentations:** the **deployed / selected** model is **Gradient Boosting** (CV recall rule). The RF plot is an **auxiliary explanatory view**, not “the deployed model’s coefficients.”

---

## Phase 5 — Evaluation (embedded in Modeling)

### Metrics reported

- **Accuracy** — easy to read but not sufficient alone.
- **Precision / Recall / F1** for **positive class** — recall matches the clinical priority stated in Phase 1.
- **ROC-AUC** — discrimination across thresholds.
- **PR-AUC (average precision)** — especially useful when prevalence differs from 50% or when comparing models under imbalance.

### Confusion matrices

- Shown for **hold-out test** predictions for each refit model.
- Interpretation guide used in the notebook:
  - **False negatives** (actual positive predicted negative) are **clinically costly** in the stated framing.

---

## Overfitting / underfitting visualizations — where are they?

### What you have now

- **Stratified CV** on the training set (variance across folds informs stability).
- **Hold-out test** evaluation (honest single-shot check).
- **Confusion matrices** on the test set.

These are **valid** pieces of evidence, but they are **not** the same as classic **learning curves** or **validation curves**.

### What “standard” overfitting diagnostics usually add

1. **Learning curve** (`sklearn.model_selection.learning_curve`): training score vs cross-validation score as training size increases. A large gap (train high, CV much lower) suggests **high variance** (often associated with overfitting). Both low suggests **underfitting** or insufficient signal.
2. **Validation curve** (`sklearn.model_selection.validation_curve`): train vs CV score vs one hyperparameter (e.g., tree depth, `C` for SVM).
3. **Train vs CV metric table** for the **selected** model (optional compact summary).

### Why they were not mandatory in your current notebook

- Many course notebooks stop at CV + test + confusion matrices.
- Learning curves require **many refits** (multiple training sizes × folds) and are sometimes omitted for runtime.

### Implemented in Modeling (after five-supervisor round on diagnostics)

`01c_Modeling_Medical.ipynb` now includes:

- **`learning_curve`** on a **`clone`** of the **deploy** estimator, with **`StratifiedKFold`** on **`X_train` / `y_train` only**, metric **average precision (PR-AUC)** — train vs CV bands for **bias–variance** and “more data?” questions.
- **Calibration curve + Brier score** on the **hold-out test** for the deploy model only — **discrimination** (AUC) vs **probability reliability**.
- **Explicit deploy policy:** **`Gradient Boosting`** is the default export target when present in the model zoo (with `cv_winner_by_mean_recall` still recorded in `medical_model_selection.json` for transparency).

---

## Five-supervisor synthesis — are you “industry standard,” and are the results “too good”?

### Industry standard (course / prototype level)

**Verdict:** The **workflow is aligned with mainstream scikit-learn practice** for tabular classification:

- stratified split,
- preprocessing fit on train only,
- **model choice by CV on training data**,
- separate **hold-out test**,
- recall-oriented metrics for a medical framing,
- exported model + metadata JSON.

**Caveat (clinical / regulated ML):** “Industry standard” for **regulated medical AI** usually adds locked protocols, extensive **leakage and cohort documentation**, **temporal** or **external** validation, calibration, subgroup analysis, and version control beyond a single course notebook. Your project is **strong at ML hygiene**; it is not automatically a **validated clinical device**.

### Second supervisor round — “foundation, not medical device” (plain English)

This thesis pipeline is **methodologically sound for a single cohort + reproducible notebook**. **Medical-device / SaMD** expectations (ISO-oriented *quality system*, risk management, software lifecycle, clinical evaluation, post-market ideas) go further: they stress **who is harmed when the model fails**, **traceable evidence**, and **governance after “go-live”** — not only hold-out AUC on one table.

| Idea | What it adds beyond your stack |
|------|--------------------------------|
| **External validation** | Another site/protocol/EHR; tests whether the frozen model + preprocessing still work when the world changes. |
| **Temporal split** | Train past / test future; random splits assume data are **exchangeable** over time — care paths and prevalence drift. |
| **Calibration** | You now report Brier + reliability plot; still **internal** — recalibration may be needed per site if probabilities drive decisions. |
| **Subgroup checks** | Overall metrics can hide worse performance by age or sex; small-N caveat applies. |
| **Protocol locking** | Pre-specify primary endpoint + threshold rule **before** “peeking” at the test set for narrative claims. |
| **Intended use** | Narrow statement: population, setting, what the score **supports** (not replaces), required human oversight. |

**Thesis “Limitations & future work” checklist (10 bullets, not legal advice):** (1) state intended use narrowly; (2) report calibration and limits of probabilities; (3) plan temporal or prospective evaluation; (4) plan external validation; (5) predefine subgroups (age/sex) if *n* allows; (6) lock primary metric + exploratory vs confirmatory wording; (7) document data lineage and index time; (8) describe human–AI workflow and override; (9) outline monitoring / drift concept; (10) map common failure modes and mitigations academically.

**Honest wording (examples):**  
- *“We followed a train–validation–test design with documented preprocessing; conclusions apply to **this benchmark dataset**, not to real-world clinical effectiveness.”*  
- *“**Not** validated externally or prospectively; **research prototype**, not a diagnostic product.”*

### Are the high scores trustworthy? (“Still treat ‘too perfect’”)

**Plausible:** Troponin and CK-MB are **strongly informative** for cardiac injury labels; your **feature importance** (RF auxiliary plot) dominated by these markers can be **biologically coherent**.

**Reviewer-ordered sanity checks** (highest impact first):

1. **Circular / definition-based leakage** — Is `Result` **in part defined** by troponin/CK-MB pathways on the **same row** or by a composite that reuses those labs? Map each feature to **permissible at index time** vs **downstream / forbidden**.
2. **Temporal leakage** — Fix an **index time** (e.g. presentation); ensure **no feature from after** the label is known unless the task is explicitly retrospective with documented windows.
3. **Duplicate patients / non-independent rows** — Unique patients vs rows; if duplicates exist, **GroupKFold** or patient-level holdout.
4. **Preprocessing leakage** — You already fit imputer/scaler on train only; avoid any future step that uses **full-data** statistics or label-dependent feature selection on the combined set.
5. **Optimistic single split / uncertainty** — CV + one test is good practice; add **bootstrap CIs by patient** (if applicable) for headline metrics; **ablation** (model without troponin/CK-MB) to show expected drop — if metrics stay “perfect,” revisit label/leakage.

**What convinces a skeptical panel:** one-page **data diagram** (index time, label, feature window); **feature inventory** (allowed vs forbidden); **patient-level** split if relevant; **ablation**; error analysis linked to collection mechanics, not only AUC.

### Presentation consistency (supervisor note)

- **Deployed model story = Gradient Boosting** + `medical_best_model.pkl` / `medical_model_selection.json`.
- **Random Forest importance** = **interpretability sidecar**, not the champion model’s internal representation.

---

## File map (quick reference)

| Phase | Notebook |
|------|----------|
| Data Understanding | `classification_Medical_data _set/01_Classification_Medical.ipynb` |
| Data Preparation | `classification_Medical_data _set/01b_DataPreparation_Medical.ipynb` |
| Modeling | `classification_Medical_data _set/01c_Modeling_Medical.ipynb` |

| Output | Location |
|--------|----------|
| Processed matrices | `classification_Medical_data _set/data/processed/medical/*.csv` |
| Preparation metadata | `classification_Medical_data _set/data/processed/medical/metadata.json` |
| Selected model | `classification_Medical_data _set/models/medical_best_model.pkl` |
| Selection record | `classification_Medical_data _set/models/medical_model_selection.json` |

---

## Suggested oral-defense “one-liner”

> We followed CRISP-DM: we explored and audited the medical table, prepared features with train-only imputation and robust scaling to avoid leakage, selected the classifier by stratified cross-validation on the training set prioritizing recall for the positive class, and reported final performance once on a held-out test set, exporting the chosen model with reproducibility metadata.

---

*Generated as project documentation for the Medical classification track. Update numeric metrics in slides from your latest notebook run if they change after re-execution.*
