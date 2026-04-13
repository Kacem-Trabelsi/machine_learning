# Contenu de Présentation (Soutenance Machine Learning)

*Note: Ce document est la version complète. Il fusionne les explications théoriques détaillées (pour votre discours et vos slides de texte) avec les marqueurs visuels (pour savoir exactement où placer vos graphiques).*

---

## 1. Introduction (Contexte & Problématique)

**Contexte :** 
L'intégration de l'Intelligence Artificielle (IA) dans le domaine de la santé représente une avancée majeure. Les hôpitaux génèrent quotidiennement d'immenses volumes de données cliniques (dossiers patients, constantes vitales, prescriptions).
**Problématique :**
Comment exploiter ces données hétérogènes (textes complexes, séries temporelles, constantes cliniques) via le Machine Learning pour :
1. Prédire les urgences critiques (Crise cardiaque).
2. Optimiser la gestion hospitalière (Durée d'hospitalisation).
3. Découvrir des profils de patients cachés (Clustering en Réanimation COVID).

---

## 2. Business Understanding et Datasets

**Tableau des BOS, DSOs et Datasets associés :**

| Dataset | BOS (Business Objective) | DSO (Data Science Objective) | Type d'Apprentissage |
| :--- | :--- | :--- | :--- |
| **1. Medical (Classification)** | Prévenir les décès par crise cardiaque en alertant les médecins instantanément. | Prédire à partir des biomarqueurs si un patient fait une crise cardiaque (1) ou non (0). | Apprentissage Supervisé (Classification Binaire) |
| **2. Hospital (Regression)** | Optimiser l'allocation des lits et aider les médecins à définir la durée des prescriptions. | Prédire avec précision la "Durée de traitement (jours)" à partir des prescriptions. | Apprentissage Supervisé (Régression Continue) |
| **3. Sirio-Libanés ICU (Clustering)** | Mieux comprendre l'évolution clinique des patients COVID pour adapter les protocoles. | Regrouper les patients selon la gravité de leurs constantes sans cible prédéfinie. | Apprentissage Non-Supervisé (Clustering Géométrique) |

> 🖼️ **[PLACEHOLDER GRAPHIQUE 1 : Traitement du volume de données - Sirio]**
> *Où le trouver :* Notebook `03_Clustering_Sirio.ipynb` (Phase 2).
> *Lequel prendre :* Le graphique en barres horizontales rouges "Top 30 Columns with Highest Missing Value Percentage".
> *Discours associé :* "Pour atteindre ces objectifs business, nous avons été confrontés à la réalité du terrain : des datasets où 50% des données vitales étaient manquantes, exigeant des protocoles de nettoyage massifs."

---

## 3. Compréhension et préparation des données (Data Prep)

**Dataset 1 : Medical Data**
- **Variables :** Cible `Heart_Attack` (0 ou 1). Variables explicatives : `Troponin`, `Systolic/Diastolic BP`, `Cholesterol`.
- **Prétraitement & Feature Engineering :** 
  - Création clinique de la variable `Pulse_Pressure` (Systolique - Diastolique).
  - Au lieu du classique StandardScaler, nous avons utilisé un **RobustScaler**. Les données cardiaques contiennent de vrais "outliers" vitaux (une tension à 200 n'est pas une erreur de frappe, c'est une anomalie médicale vitale qu'il ne faut pas écraser).
> 🖼️ **[PLACEHOLDER GRAPHIQUE 2 : Medical Analysis]**
> *Où le trouver :* Notebook `01_Classification_Medical.ipynb` (Phase 2).
> *Lequel prendre :* Un boxplot montrant la distribution de `Troponin` face à `Heart Attack`.

**Dataset 2 : Hospital Data**
- **Variables :** Cible `Duration (days)`. Explicatives : `Dosage`, texte du `Diagnosis`.
- **Prétraitement Technologique :** 
  - **Asymétrie :** La variable Dosage allait de 1g à 960g. Appliquer une Transformation Logarithmique (`np.log1p`) a permis d'écraser cette asymétrie sans détruire la donnée.
  - **NLP (Natural Language Processing) :** Transformation des diagnostics textuels complexes en matrices de "Poids de mots" grâce à l'algorithme **TF-IDF Vectorizer**.
> 🖼️ **[PLACEHOLDER GRAPHIQUE 3 : TF-IDF NLP Result]**
> *Où le trouver :* Notebook `02b_DataPreparation_Hospital.ipynb` (Phase 3).
> *Lequel prendre :* Un screenshot du DataFrame illustrant la matrice textuelle convertie en nombres décimaux (Diagnosis_chest : 0.456).

**Dataset 3 : Sirio ICU (Séries Temporelles)**
- **Prétraitement Avancé :**
  - **Imputation Longitudinale :** Remplissage des `NaNs` par `Forward-Fill / Backward-Fill` groupé par patient pour préserver la trame temporelle individuelle.
  - **Encodage Ordinal :** Transformation des quartiles d'âge textuels ("10th") en entiers purs (1, 2) pour la distance spatiale.
> 🖼️ **[PLACEHOLDER GRAPHIQUE 4 : PCA Crushing]**
> *Où le trouver :* Notebook `03b_DataPreparation_Sirio.ipynb` (Phase 3).
> *Lequel prendre :* Prenez un screenshot de la cellule affichant : `"Original Columns: 228 -> PCA Crushed Columns: 40"`.
> *Discours associé :* "Pour éviter la malédiction de la dimensionnalité en clustering, nous avons utilisé le PCA (Principal Component Analysis) pour compresser 228 variables en 40 "super-colonnes", conservant 95% de la variance."

---

## 4. Modélisation et 5. Évaluation des performances

**Choix des Métriques et Interprétations (Obligatoire) :**
- **Classification :** Pourquoi avoir choisi le **Recall (Sensibilité)** au lieu de l'Accuracy ? *Dans un contexte de crise cardiaque, un Faux Positif (alarme inutile) est gérable. Cependant, un Faux Négatif (rater une crise cardiaque) est mortel. Le modèle devait optimiser la détection maximale.*
- **Régression :** Utilisation du **RMSE** (Mesure physique de l'erreur en jours) couplé au **R2 Score** (Mesure du pourcentage de la variance que le modèle parvient à expliquer de la réalité).
- **Clustering :** Évaluation mathématique en aveugle via **l'Inertie (Méthode du Coude)** et le **Coefficient de Silhouette** (mesurant la netteté et la distance entre les groupes de gravité).

> 🖼️ **[PLACEHOLDER GRAPHIQUE 5 : Modèles de Classification]**
> *Où le trouver :* Notebook `01c_Modeling_Medical.ipynb` (Phase 4).
> *Lequel prendre :* Le graphique en barres classant les algorithmes selon leur métrique Recall.

> 🖼️ **[PLACEHOLDER GRAPHIQUE 6 : Le Clustering 3D]**
> *Où le trouver :* Notebook `03c_Modeling_Sirio.ipynb` (Phase 4).
> *Lequel prendre :* Le **3D Scatterplot** (Le cube 3D interactif).
> *Discours associé :* "Ceci est la matérialisation physique de l'apprentissage non-supervisé. Sans jamais lui donner les étiquettes ICU, l'Intelligence Artificielle a mathématiquement isolé 3 espaces cliniques de patients : le regroupement rouge (dégradation), vert (critique), et violet (stable)."

---

## 6. Benchmarking et Choix des Modèles (Conclusion)

**Tableau Comparatif & Références aux Complexités Théoriques :**

| Objectif | Le Vainqueur Déployé | Pourquoi ce choix ? (Simplicité / Robustesse) | Faiblesses |
| :--- | :--- | :--- | :--- |
| **Classification (Medical)** | **Gradient Boosting** | Le meilleur taux de Recall (93%). Il bat la régression logistique grâce à sa capacité à construire des arbres séquentiels qui minimisent l'erreur précédente de manière agressive. | Modèle lourd mathématiquement risquant le surapprentissage sans "early stopping". |
| **Régression (Hospital)** | **Random Forest Regressor** | Pertinence parfaite par rapport au problème : Les arbres de décision ingèrent exceptionnellement bien les matrices creuses pleines de zéros générées par notre algorithme NLP TF-IDF (contrairement au SVM linéaire). | Modèle statique "Boîte Noire" dont les prédictions sont dures à expliquer médicalement. |
| **Clustering (Sirio ICU)** | **K-Means (K=3)** | Simplicité algorithmique imbattable une fois que le traitement spatial (PCA) a écrasé l'espace à 40 dimensions. Robustesse absolue si les données sont bien *Mises à l'échelle Standard*. | Force l'algorithme à faire des sphères mathématiques parfaites. |

**Conclusion de Soutenance :**
Même si certains modèles retenus (comme le Random Forest) ajoutent de la complexité par rapport à des méthodes linéaires traditionnelles, l'exigence clinique quant au Recall et au traitement de champs diagnostiques textuels (NLP) rendait le sacrifice de l'interprétabilité nécessaire. Tous ces modèles ont été exportés sous formats binaires `.pkl` et sont prêts à être implémentés au sein d'une architecture Microservice API en cloud.
