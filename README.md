# houses_price

# 🏠 Estimation du Prix d'une Maison — Workshop Snowflake ML

> **Réalisé par** D'Almeida Morènikè Sharlen · Bientakonne Karambiri · Stephen Aggey

---

## Présentation

Ce workshop end-to-end démontre comment construire, évaluer et déployer un modèle de machine learning de régression de prix immobiliers directement dans l'écosystème **Snowflake**, sans sortir les données de la plateforme.

Le pipeline complet couvre :

- L'ingestion de données JSON depuis Amazon S3 via un stage Snowflake
- L'exploration et le feature engineering avec Snowpark Python
- L'entraînement et la comparaison de plusieurs modèles ML (XGBoost, Random Forest, Gradient Boosting, Régression Linéaire)
- Le stockage versionné des modèles dans le **Snowflake Model Registry**
- Le déploiement d'une application de prédiction interactive via **Streamlit in Snowflake**

---

## Architecture

```
S3 (JSON brut)
     │
     ▼
Snowflake Stage ──► HOUSE_PRICES_VARIANT (VARIANT) ──► HOUSE_PRICES (table structurée)
                                                              │
                                                              ▼
                                                  Snowpark Python — EDA & Feature Engineering
                                                              │
                                                              ▼
                                         Entraînement : XGBoost · RandomForest · GradientBoosting · LinReg
                                                              │
                                                              ▼
                                               Snowflake Model Registry
                                               (alias "prod" → XGBoost)
                                                              │
                                                              ▼
                                              Streamlit App — Estimation en temps réel
```

---

## 🗂️ Structure du projet

```
.
├── eval_data_eng_et_ML.ipynb   # Notebook principal (pipeline complet)
├── streamlit_app.py            # Application Streamlit de prédiction
└── README.md                   # Ce fichier
```

---

## Prérequis

- Un compte Snowflake avec accès à **Snowpark** et **Snowflake ML**
- Un warehouse de taille `MEDIUM` minimum (auto-suspend recommandé)
- Les permissions pour créer une base de données, un schéma et accéder à un stage S3

### Ressources Snowflake créées

| Ressource | Nom |
|---|---|
| Database | `HOUSE_PRICE` |
| Schema | `HOUSE_PRICE.ML` |
| Warehouse | `ML_WH` (MEDIUM, auto-suspend 300s) |
| Stage S3 | `HOUSE_STAGE` → `s3://logbrain-datalake/datasets/house_price/` |
| Table brute | `HOUSE_PRICES_VARIANT` |
| Table finale | `HOUSE_PRICES` |
| Table de prédictions | `HOUSE_PRICES_PREDICTIONS` |
| Model Registry (XGBoost) | `HOUSE_PRICE_XGBOOST` (alias `prod`) |
| Model Registry (RF) | `HOUSE_PRICE_RF` |

---

## Dataset

Le dataset comprend **1 090 maisons** avec les caractéristiques suivantes :

| Feature | Type | Description |
|---|---|---|
| `PRICE` | Float | Prix de vente (variable cible) |
| `AREA` | Float | Surface en m² |
| `BEDROOMS` | Int | Nombre de chambres |
| `BATHROOMS` | Int | Nombre de salles de bain |
| `STORIES` | Int | Nombre d'étages |
| `MAINROAD` | yes/no | Accès à une route principale |
| `GUESTROOM` | yes/no | Chambre d'amis |
| `BASEMENT` | yes/no | Sous-sol |
| `HOTWATERHEATING` | yes/no | Chauffe-eau |
| `AIRCONDITIONING` | yes/no | Climatisation |
| `PARKING` | Int | Nombre de places de parking |
| `PREFAREA` | yes/no | Zone privilégiée |
| `FURNISHINGSTATUS` | furnished / semi-furnished / unfurnished | État d'ameublement |

**Fourchette de prix : 87 500 € — 665 000 €** · Moyenne : ~350 000 €

---

## Pipeline ML

### 1. Préparation des features

- **Encodage binaire** : les colonnes yes/no sont converties en 0/1
- **Encodage ordinal** : `FURNISHINGSTATUS` → unfurnished=0, semi-furnished=1, furnished=2
- **Scaling** : `StandardScaler` appliqué aux 12 features brutes
- **Feature engineering** (3 features dérivées calculées après scaling) :
  - `AREA_PER_BEDROOM` = AREA / (BEDROOMS + 1)
  - `COMFORT_SCORE` = BASEMENT + HOTWATERHEATING + AIRCONDITIONING + GUESTROOM
  - `AREA_X_STORIES` = AREA × STORIES
- **Split** : 80% train / 20% test (random_state=42)

### 2. Modèles entraînés

Quatre modèles ont été comparés lors de la phase d'exploration initiale, puis XGBoost a été retenu comme modèle de production.

### 3. Optimisation

- **Cross-validation 5-fold** sur XGBoost pour estimer la stabilité du modèle
- **RandomizedSearchCV** (20 itérations, cv=5) pour optimiser les hyperparamètres du RandomForest
- **Analyse et filtrage des outliers** par méthode IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR)

---

## Analyse des performances des modèles

### Comparaison initiale (100 estimateurs, XGBoost vs Régression Linéaire)

| Métrique | XGBoost | Régression Linéaire |
|---|---|---|
| **MAE (Recall)** ↓ | **22 293 €** | 40 253 € |
| **RMSE (Precision)** ↓ | **32 694 €** | 53 985 € |
| **R² (Accuracy)** ↑ | **0.8801** | 0.6732 |

### Comparaison élargie (200 estimateurs, 4 modèles)

| Modèle | MAE (Recall) ↓ | RMSE (Precision) ↓ | R² (Accuracy) ↑ |
|---|---|---|---|
| 🥇 **XGBoost** | **11 431.52 €** | **27 621.49 €** | **0.9144** |
| 🥈 **RandomForest** | 19 252.04 € | 32 091.47 € | 0.8845 |
| 🥉 **GradientBoosting** | 29 075.79 € | 40 014.57 € | 0.8205 |
| LinearRegression | 40 253.14 € | 53 984.87 € | 0.6732 |

> XGBoost est le grand vainqueur : il devance RandomForest de **+3 points de R²** et réduit la MAE de moitié par rapport à GradientBoosting. Les modèles à base d'arbres se révèlent tous nettement supérieurs à la régression linéaire, confirmant des **relations non-linéaires** fortes entre les features et le prix.

### Interprétation des métriques (modèle de production : XGBoost, 200 estimateurs)

**MAE (Mean Absolute Error) — 11 431 €**
En moyenne, le modèle se trompe de ±11 431 € sur le prix estimé. Rapporté à un prix moyen de ~350 000 €, cela représente une erreur relative d'environ **3,3%**, ce qui est excellent pour ce type de dataset tabulaire.

**RMSE (Root Mean Squared Error) — 27 621 €**
Le RMSE est ~2,4× plus élevé que la MAE, ce qui révèle quelques erreurs ponctuellement importantes sur des biens atypiques. Ce ratio MAE/RMSE est caractéristique d'un modèle bien ajusté qui gère correctement la majorité des cas mais reste sensible aux valeurs extrêmes.

**R² — 0.9144**
Le modèle explique **91,4% de la variance** des prix de vente. C'est un excellent score pour un dataset immobilier de 1 090 maisons. Les 8,6% restants correspondent à des facteurs non capturés : localisation précise, état du bien, conjoncture du marché.

### Stabilité — Cross-Validation 5-fold (XGBoost, 200 estimateurs)

```
R² moyen (5-fold CV) : 0.5433 ± 0.6575
```

> ⚠️ **Signal d'alerte important** : le R² moyen en cross-validation (0.54) est très inférieur au R² sur le test set (0.91), avec un écart-type très élevé (±0.66). Cela indique une **forte variance entre les folds** et un probable **sur-apprentissage** du modèle sur certains splits. Ce phénomène est probablement amplifié par la taille modeste du dataset (1 090 lignes) : certains folds héritent de distributions de prix très différentes.

### Analyse des outliers

```
Outliers détectés : 24 (2.2%)
Données après filtre IQR : 1 066 lignes
```

24 maisons (2,2% du dataset) ont été identifiées comme outliers par la méthode IQR (Q1 − 1,5×IQR / Q3 + 1,5×IQR). Leur retrait ramène le dataset à 1 066 observations propres, utilisées pour les modèles optimisés.

### Optimisation des hyperparamètres : RandomForest (RandomizedSearchCV)

```
Meilleurs paramètres : {'n_estimators': 300, 'min_samples_split': 2,
                        'min_samples_leaf': 1, 'max_depth': None}
R² optimisé RF : 0.8849
```

La recherche aléatoire (20 itérations, 5-fold CV, 100 fits au total) confirme que le RandomForest optimisé atteint R² = 0.8849, proche de sa version par défaut mais sans dépasser XGBoost (0.9144). XGBoost reste le choix de production.

### Features les plus importantes (XGBoost)

D'après les importances calculées par XGBoost et les coefficients de la régression linéaire, les variables les plus influentes sur le prix sont :

1. **AREA** — La surface est le facteur dominant, confirmé par les deux approches
2. **BATHROOMS** — Fortement corrélé avec le standing de la maison
3. **STORIES** — Plus il y a d'étages, plus le prix augmente significativement
4. **AIRCONDITIONING** — Équipement premium à fort impact
5. **PREFAREA** — La zone privilégiée apporte une prime de prix notable
6. **PARKING** — Chaque place supplémentaire valorise le bien

À l'inverse, `HOTWATERHEATING` et `GUESTROOM` ont un impact marginal sur le prix.

### Points forts et limites

**Points forts :**
- R² = 0.9144 avec seulement 13 features brutes est un excellent résultat
- Aucune donnée manquante dans le dataset (0 NaN après nettoyage)
- Le pipeline de feature engineering (COMFORT_SCORE, AREA_PER_BEDROOM, AREA_X_STORIES) améliore la capacité prédictive
- MAE de 11 431 € ≈ 3,3% d'erreur relative — précision très satisfaisante pour un estimateur immobilier
- Seulement 24 outliers (2,2%) dans le dataset — données de bonne qualité

**Limites & axes d'amélioration :**
- ⚠️ **Instabilité en cross-validation** : R² CV = 0.54 ± 0.66 vs R² test = 0.91 → risque de sur-apprentissage sur certains splits. Pistes : `RepeatedKFold`, régularisation XGBoost (`reg_alpha`, `reg_lambda`), augmentation du dataset
- Dataset de 1 090 maisons uniquement, un dataset plus large améliorerait la généralisation et réduirait la variance inter-folds
- Absence de données géographiques précises (quartier, ville, coordonnées GPS), probablement le facteur le plus explicatif du prix manquant
- Le RMSE (~2,4× la MAE) révèle une sensibilité résiduelle aux biens très atypiques malgré le filtrage IQR

---

## Déploiement : Snowflake Model Registry

Les modèles sont versionnés automatiquement avec un timestamp (`V{YYYYMMDD_HHMMSS}`) et enregistrés dans le Snowflake Model Registry avec :

- Les métriques (R², MAE, RMSE) attachées à chaque version
- Un alias `prod` automatiquement rotationné vers la dernière version
- Une purge automatique des anciennes versions (conservation des 3 dernières)
- Le support des deux plateformes cibles : `WAREHOUSE` et `SNOWPARK_CONTAINER_SERVICES`

---

## Application Streamlit

L'application `streamlit_app.py` offre une interface de prédiction en temps réel directement dans Snowflake.

### Fonctionnalités

- **3 sections de saisie** : Superficie & Structure · Équipements · Prestations
- **Résumé live** des caractéristiques saisies avant estimation
- **Estimation XGBoost** avec prix en euros et prix au m²
- **Positionnement marché** : barre de progression entre le min (87 500 €) et le max (665 000 €) du dataset
- **Métriques clés** : chambres, surface, étages, prix/m²
- **Détail des features** saisies exportable

### Lancement

L'app est déployée nativement dans Snowflake. Elle charge automatiquement :
1. Le scaler `StandardScaler` ajusté sur les données de production
2. Le modèle XGBoost via `Registry.get_model("HOUSE_PRICE_XGBOOST").version("prod")`

---

## 👥 Auteurs

| Nom | Rôle |
|---|---|
| D'Almeida Morènikè Sharlen | MBA2 MBDIA - Janvier 2026 |
| Bientakonne Karambiri | MBA2 MBDIA - Janvier 2026 |
| Stephen Aggey | MBA2 MBDIA - Janvier 2026 |

---

## Technologies utilisées

- **Snowflake** : Data Cloud, Snowpark, Model Registry, Streamlit in Snowflake
- **Python** : snowpark, scikit-learn, xgboost, pandas, matplotlib, seaborn
- **ML** : XGBoost, RandomForest, GradientBoosting, LinearRegression, StandardScaler, RandomizedSearchCV
- **Storage** : Amazon S3 (données source), Snowflake Tables (données structurées + prédictions)
