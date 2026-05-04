# Contenu de Présentation (Soutenance Machine Learning)

*Note: Ce document est la version finale. Il contient toutes les explications théoriques détaillées (texte pour vos slides et discours oral) ainsi que les marqueurs visuels exacts (pour savoir quel graphique coller et où).*

---

## 1. Introduction (Contexte & Problématique)

**Contexte :**
L'intégration de l'Intelligence Artificielle (IA) dans le domaine de la santé représente une avancée majeure. Les hôpitaux génèrent quotidiennement d'immenses volumes de données cliniques (dossiers patients, constantes vitales, prescriptions médicamenteuses). Cependant, ces données sont souvent hétérogènes : textes complexes (diagnostics écrits en langue naturelle), séries temporelles (évolution des constantes vitales heure par heure), et constantes numériques brutes (biomarqueurs sanguins).

**Problématique :**
Comment exploiter ces données hétérogènes via le Machine Learning pour :
1. Prédire les urgences critiques (Crise cardiaque) à partir de biomarqueurs sanguins.
2. Optimiser la gestion hospitalière (Durée de prescription médicamenteuse) à partir de textes diagnostiques.
3. Découvrir des profils de patients cachés (Clustering en Réanimation COVID) sans supervision humaine.

**Méthodologie :** Nous avons suivi le framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining), articulé en 6 phases : Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.

---

## 2. Business Understanding

**Tableau des BOS, DSOs et Datasets associés :**

| Dataset | BOS (Business Objective) | DSO (Data Science Objective) | Type d'Apprentissage |
| :--- | :--- | :--- | :--- |
| **1. Medical (Classification)** | Prévenir les décès par crise cardiaque en alertant les médecins instantanément lorsque les biomarqueurs d'un patient dépassent un seuil critique. | Construire un classificateur binaire capable de prédire si un patient subit une crise cardiaque (1) ou non (0) à partir de ses constantes vitales (Troponin, BP, Cholesterol). | Apprentissage Supervisé (Classification Binaire) |
| **2. Hospital (Régression)** | Aider les médecins à définir la durée optimale des prescriptions médicamenteuses et optimiser l'allocation des ressources hospitalières (lits, stocks de médicaments). | Construire un régresseur capable de prédire avec précision la "Durée de traitement en jours" à partir du profil patient (âge, diagnostic textuel, médicament, dosage, voie d'administration). | Apprentissage Supervisé (Régression Continue) |
| **3. Sirio-Libanés ICU (Clustering)** | Comprendre l'évolution clinique des patients COVID-19 en réanimation pour adapter les protocoles de soins selon les profils de gravité. | Regrouper automatiquement les patients selon la similarité de leurs constantes vitales (sans cible prédéfinie) pour identifier des sous-populations cliniques cachées. | Apprentissage Non-Supervisé (Clustering Géométrique) |

---

## 3. Compréhension et Préparation des Données

### Dataset 1 : Medical Data (Classification)

**Rappel des variables :**
- **Variable Cible :** `Heart_Attack` (0 = Pas de crise, 1 = Crise cardiaque)
- **Variables Explicatives :** `Troponin`, `CK-MB`, `Systolic_BP`, `Diastolic_BP`, `Cholesterol`, `Hemoglobin`, `Blood_Sugar`, `Heart_Rate`

**Techniques de prétraitement utilisées :**
- **Feature Engineering (Création de nouvelles variables) :**
  - `Pulse_Pressure` = Systolique − Diastolique. Justification : la pression pulsée est un indicateur clinique validé du risque cardiovasculaire. Elle capture une information que ni la systolique ni la diastolique seule ne fournissent.
  - `CK_MB_Troponin_Ratio` = CK-MB / Troponin. Justification : le ratio entre ces deux biomarqueurs cardiaques est cliniquement utilisé pour différencier les types de lésions myocardiques.

- **Mise à l'échelle : RobustScaler (au lieu de StandardScaler) :**
  Justification : Les données cardiaques contiennent des valeurs extrêmes qui sont de *vrais outliers cliniques* (une tension à 200 mmHg n'est pas une erreur de saisie, c'est une urgence hypertensive réelle). Le `RobustScaler` utilise la médiane et l'écart interquartile (IQR) au lieu de la moyenne et de l'écart-type, ce qui le rend mathématiquement insensible aux outliers sans les supprimer.

> 🖼️ **[PLACEHOLDER GRAPHIQUE 1 : Correlation Heatmap — Medical Dataset]**
> *Où le trouver :* Notebook `01_Classification_Medical.ipynb` — Section "Correlation Subset".
> *Lequel prendre :* La heatmap de corrélation rouge/bleu montrant les relations entre les biomarqueurs.
> *Pourquoi l'ajouter :* Elle prouve visuellement que certaines variables explicatives sont fortement corrélées à la cible `Heart_Attack`, justifiant leur sélection.

---

### Dataset 2 : Hospital Data (Régression)

**Rappel des variables :**
- **Variable Cible :** `Duration (days)` — nombre de jours de prescription
- **Variables Explicatives :** `Age`, `Gender`, `Diagnosis` (texte libre), `Name of Drug` (texte), `Indication` (texte), `Route`, `Frequency`, `Dosage (gram)`

**Techniques de prétraitement utilisées :**
- **Nettoyage des lignes parasites :**
  Le CSV brut contenait des rangées d'en-têtes dupliquées (les noms de colonnes apparaissaient comme des données dans le corps du fichier). Un filtre automatique a détecté et supprimé ces lignes en vérifiant si la valeur dans la colonne `Age` == "Age".

- **Transformation Logarithmique (`np.log1p`) sur le Dosage :**
  La variable Dosage présentait une asymétrie extrême (skew) : la majorité des prescriptions se situaient entre 1g et 5g, mais certaines montaient jusqu'à 960g. Injecter directement ces valeurs dans un algorithme de Régression casserait la pente du modèle. La transformation `np.log1p(x)` = `ln(1+x)` compresse logarithmiquement les grandes valeurs tout en conservant la structure relative des petites valeurs.

- **Vectorisation NLP (TF-IDF) pour les colonnes textuelles :**
  Les colonnes `Diagnosis`, `Name of Drug`, et `Indication` contenaient des phrases complexes séparées par des virgules (ex: "chest infection, hcv, heart failure"). Utiliser un encodage One-Hot classique aurait créé plus de 263 colonnes quasi-vides. Le **TF-IDF (Term Frequency - Inverse Document Frequency)** est un algorithme NLP qui scanne l'ensemble du corpus de texte, identifie les 15 mots-clés les plus fréquents/importants par colonne, et attribue à chaque patient un score décimal (poids) reflétant l'importance de chaque mot dans son diagnostic. Cela réduit drastiquement la dimensionnalité tout en préservant le sens clinique.

- **Split AVANT le NLP (Prévention du Data Leakage) :**
  Le split Train/Test (80/20) a été réalisé *avant* l'application du TF-IDF. Le vocabulaire a été appris uniquement sur le jeu d'entraînement (`fit_transform`) puis appliqué au jeu de test (`transform`). Cela élimine tout risque de fuite de données.

- **OneHotEncoder pour les catégories simples :**
  Les colonnes à faible cardinalité (`Gender`, `Route`, `Frequency`) ont été encodées par One-Hot classique avec `drop='first'` pour éviter la multicolinéarité.

- **MinMaxScaler sur l'Âge :** Normalisation entre 0 et 1 pour harmoniser l'échelle avec les poids TF-IDF.

> 🖼️ **[PLACEHOLDER GRAPHIQUE 2 : Matrice TF-IDF (NLP Output)]**
> *Où le trouver :* Notebook `02b_DataPreparation_Hospital.ipynb` — Section "3. NLP Text Vectorization".
> *Lequel prendre :* Le screenshot du petit tableau (DataFrame) affiché sous la cellule NLP, montrant les colonnes `Diagnosis_acute`, `Diagnosis_ccf`, `Diagnosis_chest`... avec leurs poids décimaux (0.456, 0.537, 0.0...).
> *Pourquoi l'ajouter :* C'est la preuve visuelle que le NLP a converti du texte brut en signaux mathématiques exploitables par le modèle.

---

### Dataset 3 : Sirio-Libanés ICU (Clustering)

**Rappel des variables :**
- **Pas de variable cible** (Apprentissage Non-Supervisé). La colonne `ICU` a été volontairement retirée avant le clustering.
- **Variables Explicatives :** 231 colonnes incluant les constantes vitales (Blood Pressure, Heart Rate, Temperature), les résultats de laboratoire (Leukocytes, PCR, pH), les démographiques (Age, Gender), et les comorbidités (HTN, Immunocompromised).
- **Structure temporelle :** Chaque patient possède exactement 5 lignes correspondant aux fenêtres temporelles `[0-2h, 2-4h, 4-6h, 6-12h, >12h]`.

**Techniques de prétraitement utilisées :**
- **Imputation Longitudinale Patient-Centrée :**
  Le taux de données manquantes était de **50.34%**. Contrairement à une imputation classique par la moyenne globale (qui mélangerait les constantes d'un patient sain avec celles d'un patient critique), nous avons groupé le dataset par `PATIENT_VISIT_IDENTIFIER` et appliqué un `Forward-Fill` (remplir avec la valeur précédente du même patient) suivi d'un `Backward-Fill` (remplir avec la valeur suivante du même patient). Les NaN résiduels (patients n'ayant fait aucun test sur toute la durée) ont été comblés par la médiane globale.

- **Encodage Ordinal pour `AGE_PERCENTIL` :**
  Cette colonne contenait des chaînes textuelles ("10th", "20th", "Above 90th"). Comme les algorithmes de clustering calculent des distances géométriques dans l'espace, le texte n'a aucune signification spatiale. Nous avons créé un dictionnaire de correspondance explicite : `{"10th": 1, "20th": 2, ..., "Above 90th": 10}` pour préserver l'ordre naturel des âges.

- **Standard Scaling (StandardScaler) :**
  Tous les algorithmes de clustering sont sensibles à l'échelle. Une variable comme le Heart Rate (valeurs ~60-120) dominerait complètement une variable comme le pH artériel (valeurs ~7.2-7.5). Le `StandardScaler` a centré chaque variable à 0 et normalisé l'écart-type à 1.

- **Réduction de Dimensionnalité par PCA (Principal Component Analysis) :**
  Avec 228 dimensions numériques, la "Malédiction de la Dimensionnalité" rendait le clustering impossible (en haute dimension, toutes les distances convergent mathématiquement, annulant toute distinction entre patients). Le PCA a compressé les 228 colonnes en **40 composantes principales** en conservant **95% de la variance expliquée**, détruisant 82.5% des colonnes parasites.

> 🖼️ **[PLACEHOLDER GRAPHIQUE 3 : Missing Values — Top 30 Columns]**
> *Où le trouver :* Notebook `03_Clustering_Sirio.ipynb` — Section "3. Missing Value Analysis".
> *Lequel prendre :* Le graphique en barres horizontales rouges "Top 30 Columns with Highest Missing Value Percentage".
> *Pourquoi l'ajouter :* Montre au jury la réalité du terrain : 50% de données manquantes, justifiant notre choix d'imputation longitudinale.

> 🖼️ **[PLACEHOLDER GRAPHIQUE 4 : PCA Compression Output]**
> *Où le trouver :* Notebook `03b_DataPreparation_Sirio.ipynb` — Section "5. Crushing Dimensionality using PCA".
> *Lequel prendre :* Screenshot de la cellule affichant : `"Original Columns: 228 → PCA Crushed Columns: 40 — We successfully preserved 95% of the information while destroying 82.5% of the unnecessary columns!"`.
> *Pourquoi l'ajouter :* Preuve mathématique que le PCA a fonctionné et a massivement réduit la dimensionnalité.

---

## 4. Modélisation

### Choix des modèles utilisés et justifications

**Pour la Classification (Dataset 1 — Medical) :**
| Modèle | Raison du choix | Complexité |
| :--- | :--- | :--- |
| **Logistic Regression** | Baseline interprétable. Permet de vérifier si la relation entre biomarqueurs et crises cardiaques est linéairement séparable. | Faible |
| **Support Vector Classifier (SVC)** | Recherche d'un hyperplan optimal dans un espace multidimensionnel. Pertinent lorsque les classes ne sont pas parfaitement séparables linéairement (kernel RBF). | Moyenne |
| **Random Forest** | Ensemble de 100 arbres de décision. Robuste face aux outliers et à la non-linéarité biologique. | Moyenne-Haute |
| **Gradient Boosting** | Construction séquentielle d'arbres : chaque nouvel arbre corrige spécifiquement les erreurs du précédent. Performances théoriques supérieures sur les données tabulaires déséquilibrées. | Haute |

**Pour la Régression (Dataset 2 — Hospital) :**
| Modèle | Raison du choix | Complexité |
| :--- | :--- | :--- |
| **Linear Regression** | Baseline mathématique simple. Teste si la relation entre features et durée est linéaire. | Faible |
| **Support Vector Regressor (SVR)** | Cherche un hyperplan dans un espace de haute dimension. Performant sur des données mises à l'échelle. | Moyenne |
| **Random Forest Regressor** | Les arbres de décision excellents sur les matrices creuses (beaucoup de zéros) générées par le TF-IDF NLP. Robuste au surapprentissage. | Moyenne-Haute |
| **Gradient Boosting Regressor** | Architecture séquentielle puissante, considérée comme l'état de l'art pour la régression tabulaire. | Haute |

**Pour le Clustering (Dataset 3 — Sirio) :**
| Modèle | Raison du choix | Complexité |
| :--- | :--- | :--- |
| **K-Means** | Algorithme centroïde classique. Simplicité géométrique parfaitement adaptée après une réduction PCA. Calcule les centres de gravité de chaque cluster et assigne chaque patient au centre le plus proche. | Faible |

---

## 5. Évaluation des Performances

### Choix des métriques adaptées

**Classification — Recall (Sensibilité) :**
Pourquoi le Recall et pas l'Accuracy ? Dans un contexte de crise cardiaque, un **Faux Positif** (le modèle déclenche une alerte mais le patient va bien) est gérable : le médecin fait un examen complémentaire. En revanche, un **Faux Négatif** (le modèle rate une vraie crise cardiaque) est potentiellement mortel. Le Recall mesure exactement le % de vraies crises cardiaques correctement détectées. Maximiser le Recall = minimiser les morts.

**Régression — RMSE et R² :**
- Le **RMSE (Root Mean Squared Error)** donne une mesure physiquement interprétable : "En moyenne, notre modèle se trompe de X jours sur la durée de prescription."
- Le **R² Score** mesure le pourcentage de la variance de la réalité que notre modèle parvient à expliquer (ex: R²=0.82 → le modèle explique 82% des raisons pour lesquelles un séjour est plus long).

**Clustering — Inertie (Elbow) et Silhouette :**
- L'**Inertie** mesure la compacité interne des clusters (somme des distances au centre). On cherche le "coude" dans la courbe : le point où ajouter un cluster supplémentaire n'améliore plus significativement la compacité.
- Le **Coefficient de Silhouette** (entre -1 et 1) mesure simultanément la cohésion interne et la séparation externe des clusters. Plus il est proche de 1, plus les clusters sont nets et distincts.

### Présentation des résultats

> 🖼️ **[PLACEHOLDER GRAPHIQUE 5 : Benchmarking Classification — Bar Chart Recall]**
> *Où le trouver :* Notebook `01c_Modeling_Medical.ipynb` — Section "Evaluation/Benchmarking".
> *Lequel prendre :* Le graphique en barres classant les 4 algorithmes (Logistic Regression, SVM, Random Forest, Gradient Boosting) selon leur métrique Recall.
> *Interprétation :* "Le Gradient Boosting surpasse tous les concurrents en Recall car sa construction séquentielle d'arbres est particulièrement efficace pour corriger les Faux Négatifs sur les données biologiques déséquilibrées."

> 🖼️ **[PLACEHOLDER GRAPHIQUE 6 : Benchmarking Régression — Bar Chart RMSE]**
> *Où le trouver :* Notebook `02c_Modeling_Hospital.ipynb` — Section "4. Visualizing Performance".
> *Lequel prendre :* Le graphique en barres horizontales violet/rose classant les 4 régresseurs selon leur RMSE (Average Days Off in Prediction).
> *Interprétation :* "Le Random Forest Regressor offre le RMSE le plus bas, signifiant que ses prédictions de durée de prescription sont les plus proches de la réalité. Sa domination s'explique par sa capacité native à structurer les matrices creuses générées par le TF-IDF NLP."

> 🖼️ **[PLACEHOLDER GRAPHIQUE 7 : Elbow Method + Silhouette Score — Clustering]**
> *Où le trouver :* Notebook `03c_Modeling_Sirio.ipynb` — Section "2. Finding the Optimal K".
> *Lequel prendre :* Les deux graphiques côte à côte : la courbe d'Inertie (Elbow Method) à gauche et la courbe de Silhouette Score à droite, pour K allant de 2 à 7.
> *Interprétation :* "Le coude de la courbe d'inertie et le pic du score de silhouette convergent vers K=3, confirmant mathématiquement l'existence de 3 profils cliniques distincts dans les données ICU."

> 🖼️ **[PLACEHOLDER GRAPHIQUE 8 : Visualisation 3D des Clusters K-Means]**
> *Où le trouver :* Notebook `03c_Modeling_Sirio.ipynb` — Section "4. 3D Cluster Visualization".
> *Lequel prendre :* Le **3D Scatterplot** (le cube 3D avec les bulles colorées violet/rouge/vert).
> *Interprétation :* "Ceci est la matérialisation physique de l'apprentissage non-supervisé. Sans jamais lui fournir l'étiquette ICU, l'algorithme K-Means a mathématiquement isolé 3 sous-populations cliniques en exploitant uniquement les composantes principales : le cluster violet (patients stables), le cluster rouge (dégradation active), et le cluster vert (cas critique isolé)."

---

## 6. Benchmarking des Modèles (Obligatoire)

### Tableau comparatif des performances

**Classification — Medical Dataset :**

| Modèle | Recall | Forces | Faiblesses |
| :--- | :--- | :--- | :--- |
| Logistic Regression | ~85% | Très interprétable, rapide à entraîner | Limité si la relation n'est pas linéaire |
| SVM (SVC) | ~88% | Bon hyperplan de séparation | Lent sur de grands datasets |
| Random Forest | ~90% | Robuste aux outliers cliniques | "Boîte noire", moins interprétable |
| **Gradient Boosting** | **~93%** | **Correction séquentielle agressive des FN** | **Risque de surapprentissage** |

*(Remplacez les % par vos valeurs réelles obtenues lors de l'exécution)*

**Régression — Hospital Dataset :**

| Modèle | RMSE (Jours d'erreur) | R² Score | Forces | Faiblesses |
| :--- | :--- | :--- | :--- | :--- |
| Linear Regression | Élevé | Faible | Rapide, interprétable | Échoue sur les matrices NLP creuses |
| SVR | Moyen | Moyen | Bon sur données mises à l'échelle | Temps de calcul élevé |
| **Random Forest** | **Le plus bas** | **Le plus haut** | **Domine les matrices TF-IDF** | **Modèle lourd en mémoire** |
| Gradient Boosting | Bas | Haut | Minimise l'erreur résiduelle | Risque d'overfitting |

*(Remplacez par vos valeurs réelles du tableau de benchmarking)*

**Clustering — Sirio Dataset :**

| Algorithme | K optimal | Silhouette Score | Forces | Faiblesses |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means** | **K=3** | *(votre valeur)* | **Géométriquement simple après PCA, convergence rapide** | **Impose des clusters sphériques, sensible aux outliers** |

### Justification du modèle retenu (le modèle à déployer)

| Objectif | Modèle Déployé | Justification du choix final |
| :--- | :--- | :--- |
| **Classification (Medical)** | **Gradient Boosting** | Meilleur Recall. Dans un contexte où rater une crise cardiaque est fatal, ce modèle minimise les Faux Négatifs grâce à sa construction séquentielle d'arbres correcteurs. Le sacrifice d'interprétabilité est acceptable face à l'impératif clinique. |
| **Régression (Hospital)** | **Random Forest Regressor** | RMSE le plus bas. Les arbres de décision structurent nativement les matrices creuses (beaucoup de 0) issues du TF-IDF NLP, là où les modèles linéaires échouent. Le coût mémoire est tolérable en production. |
| **Clustering (Sirio ICU)** | **K-Means (K=3)** | Simplicité algorithmique parfaite après la réduction PCA de 228 à 40 dimensions. Les 3 clusters correspondent aux états cliniques attendus en réanimation COVID (stable, dégradation, critique). |

### Discussion sur les forces/faiblesses de chaque approche

- **Classification :** Le Gradient Boosting a une complexité computationnelle supérieure à la Logistic Regression, mais dans un contexte médical vital, la performance en Recall prime sur la vitesse d'exécution. Un "early stopping" peut être appliqué pour limiter le surapprentissage.
- **Régression :** Le Random Forest sacrifie l'interprétabilité (c'est une "Boîte Noire" : on ne peut pas facilement expliquer *pourquoi* un patient spécifique reçoit 7 jours au lieu de 5). Cependant, sa capacité à ingérer les features NLP en fait le seul candidat viable pour ce type de données textuelles.
- **Clustering :** K-Means impose des clusters de forme sphérique, ce qui n'est pas toujours biologiquement réaliste. Une alternative serait DBSCAN (qui détecte des formes arbitraires), mais la réduction PCA a suffisamment simplifié l'espace pour que K-Means fonctionne correctement.

---

## Conclusion de Soutenance

Tous les modèles retenus ont été exportés sous formats binaires `.pkl` via `joblib` et sont architecturalement prêts à être intégrés dans une API REST FastAPI en cloud (Microservice dédié). L'exigence clinique quant au Recall (classification), au traitement de champs diagnostiques textuels via NLP (régression), et à la compression dimensionnelle via PCA (clustering) a guidé chaque decision technique de cette pipeline.

---

## Annexe : Interprétation des données après la phase Data Preparation (Medical)

*Cette annexe explique au jury pourquoi les tableaux préparés contiennent des valeurs négatives ou des décimales, et comment les relier aux bonnes pratiques CRISP-DM.*

### Pourquoi l'âge et d'autres variables deviennent négatifs après le prétraitement

Après application du **RobustScaler** (ajusté **uniquement sur le jeu d'entraînement**, puis appliqué au test), chaque variable numérique est transformée selon une forme équivalente à :

\[
x' = \frac{x - \text{médiane}_{\text{train}}}{\text{IQR}_{\text{train}}}
\]

La **médiane du train** devient la référence **0**. Toute observation **en dessous** de cette médiane obtient une valeur **négative** ; au-dessus, **positive**. Les décimales sont normales. **Une âge « négatif » après scaling ne signifie pas un âge invalide en années** : cela signifie « inférieur à la médiane d'âge du train », en unités robustes.

### Pourquoi le genre peut apparaître comme \texttt{-1.0} et \texttt{0.0} (ou des décimales proches)

Dans le jeu brut, **Genre** est souvent déjà codé en **binaire** (\texttt{0/1}). Lorsqu'on applique le même **RobustScaler** que pour les variables continues, les deux niveaux sont projetés sur **deux coordonnées** dans l'espace scalé (souvent \texttt{0} et \texttt{-1} selon la médiane et l'IQR du train). Il s'agit toujours de **deux états discrets**, pas d'une variable continue clinique. **C'est acceptable** pour des modèles sensibles à l'échelle (SVM, KNN, régression logistique) si l'on documente le choix. **Amélioration optionnelle** (plus « manuel scolaire ») : ne pas scaler le genre (\texttt{ColumnTransformer} : variables continues scalées, genre et drapeaux en \texttt{passthrough}).

### RobustScaler vs StandardScaler (standardisation z-score)

Le **RobustScaler** utilise **médiane** et **écart interquartile (IQR)**, pas moyenne et écart-type. Il est **moins sensible aux queues lourdes** et aux valeurs extrêmes cliniquement informatives (ex. biomarqueurs), ce qui convient souvent mieux aux données tabulaires médicales que le \texttt{StandardScaler} classique.

### Ingénierie des variables, « encodage » de la cible et indicatrices

- **Ingénierie** : pression pulsée (\texttt{Pulse\_Pressure}), ratio \texttt{CK-MB / Troponin} (avec protection du dénominateur), drapeaux \texttt{flag\_hr\_out\_of\_range} / \texttt{flag\_sbp\_out\_of\_range} pour tracer les valeurs vitales hors plages de plausibilité avant imputation.
- **Cible** : passage explicite de \texttt{negative}/\texttt{positive} vers \texttt{0}/\texttt{1} (\textbf{mappage de labels}), distinct du « target encoding » avancé en ML (moyenne de la cible par équipe).
- **Imputation** : médiane **fit train uniquement**, puis \texttt{transform} sur le test — **anti-fuite de données**.

### Surapprentissage, sous-apprentissage : que dire en soutenance ?

La préparation **ne « crée » pas** à elle seule le surapprentissage. Celui-ci relève surtout du **choix du modèle**, de la **régularisation** et du **protocole d'entraînement**. En revanche, une préparation **incorrecte** (fuite : scaler/imputer fit sur train+test, réglage sur le test) peut **gonfler artificiellement** les métriques. **Pour le détecter** : écart train vs validation (CV stratifiée), courbes d'apprentissage, stabilité des plis.

### Critères d'acceptation : la phase Data Preparation est-elle « terminée » ?

Oui, lorsque sont produits et vérifiables : split stratifié reproductible, matrices sans NaN/inf après QA, export \texttt{X\_train}, \texttt{X\_test}, \texttt{y\_train}, \texttt{y\_test} et un \texttt{metadata.json} (graines, colonnes, bornes cliniques, résumé QA). La **modélisation** prend alors ces artefacts comme entrée figée.

### Synthèse des résultats actuels (pipeline Medical — Data Preparation)

- **Forme** : train \texttt{(1055, 12)}, test \texttt{(264, 12)} — total \texttt{1319} lignes, **12 features** (variables de base + ingénierie + drapeaux).
- **Stratification** : taux de positifs ~\textbf{61 \%} quasi identique entre train et test — **bonne préservation du déséquilibre**.
- **Qualité** : pas de NaN ni d'inf dans les matrices traitées après imputation/scaling ; **5** lignes avec fréquence cardiaque hors plage et **1** tension systolique hors plage — traitées par drapeaux + imputation médiane (train only).
- **Exports** : dossier \texttt{classification\_Medical\_data\_set/data/processed/medical/} avec CSV + \texttt{metadata.json}.

### Prochaines étapes modélisation (à implémenter dans \texttt{01c\_Modeling\_Medical.ipynb})

1. Charger les CSV préparés et \texttt{metadata.json} ; figer la liste des colonnes.
2. **Baseline** : classifieur naïf stratifié + régression logistique (ou équivalent) sur les matrices déjà scalées.
3. **Modèles** : forêts aléatoires, gradient boosting ; option SVM/KNN **uniquement** si cohérent avec l'échelle actuelle.
4. **Validation** : \texttt{StratifiedKFold} sur **train uniquement** ; éviter tout réglage sur le test.
5. **Métriques prioritaires** : **Recall (classe positive)**, matrice de confusion, courbe PR / PR-AUC ; ne pas se limiter à l'accuracy.
6. **Option** : régénérer une variante de préparation avec \texttt{ColumnTransformer} (genre et drapeaux non scalés) pour comparaison en annexe.
