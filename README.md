# 🏠 Prévision de Prix Immobiliers — DVF Open Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Modélisation et prévision des prix de l'immobilier en France à partir des **Demandes de Valeurs Foncières (DVF)** publiées en open data sur [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/). Le pipeline complet nettoie plus d'un million de transactions, construit des variables géographiques et temporelles, entraîne un modèle **XGBoost**, et expose les prédictions via un **dashboard Streamlit** interactif.

---

## 📋 Sommaire

- [Aperçu](#-aperçu)
- [Architecture du projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline de données](#-pipeline-de-données)
- [Modélisation](#-modélisation)
- [Dashboard Streamlit](#-dashboard-streamlit)
- [Résultats](#-résultats)
- [Reproductibilité](#-reproductibilité)
- [Licence](#-licence)

---

## 🎯 Aperçu

Ce projet illustre un **workflow data science de bout en bout** appliqué au marché immobilier français :

- **Nettoyage et EDA** de plus d'un million de transactions DVF (2019-2024)
- **Ingénierie de variables** géographiques (département, région, prix médians locaux), typologiques (type de bien, surface, nombre de pièces) et temporelles (saisonnalité, tendance annuelle)
- **Modèle XGBoost** entraîné avec validation croisée temporelle, évalué par **RMSE** et **R²**
- **Dashboard Streamlit** permettant à un utilisateur d'estimer le prix d'un bien selon sa région, son type et ses caractéristiques

### Stack technique

| Domaine | Outils |
|---|---|
| Ingestion & nettoyage | Python, pandas, DuckDB / SQL |
| Feature engineering | pandas, geopandas, scikit-learn |
| Modélisation | XGBoost, scikit-learn |
| Visualisation | Plotly, matplotlib, seaborn |
| Déploiement | Streamlit |
| Qualité du code | pytest, ruff, black |

---

## 📁 Architecture du projet

```
dvf-forecast/
├── data/
│   ├── raw/                  # Fichiers DVF bruts (CSV depuis data.gouv.fr)
│   └── processed/            # Données nettoyées & enrichies (parquet)
├── src/
│   ├── __init__.py
│   ├── config.py             # Constantes, chemins, hyperparamètres
│   ├── data_loader.py        # Téléchargement multi-années DVF
│   ├── cleaning.py           # Nettoyage + winsorisation adaptative par dept
│   ├── enrich.py             # K-means géographique + target encoding anti-fuite
│   ├── features.py           # Feature engineering (fit/transform split)
│   ├── train.py              # XGBoost + CV temporelle anti-fuite
│   ├── evaluate.py           # Métriques RMSE / R² / MAPE par segment
│   └── predict.py            # Interface de prédiction
├── notebooks/
│   ├── 01_exploration.ipynb  # EDA initiale
│   ├── 02_features.ipynb     # Conception des variables
│   ├── 03_modeling.ipynb     # Itérations modèles
│   └── 04_diagnostic.ipynb   # Diagnostic complet (14 sections)
├── streamlit_app/
│   └── app.py                # Dashboard interactif
├── models/                   # Modèles + artefacts (.joblib)
├── tests/                    # Tests unitaires (17 tests)
├── sql/queries.sql           # Requêtes DuckDB d'agrégation
├── requirements.txt
├── Makefile / run.bat        # Orchestration (Unix / Windows)
├── .github/workflows/ci.yml  # CI tests automatisés
└── README.md
```

---

## ⚙️ Installation

### Prérequis

- Python **3.10+** (y compris **3.13**) — Windows, macOS, Linux
- ~ 4 Go de RAM disponibles pour le traitement (DVF complet)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/rayenaissa/dvf-forecast.git
cd dvf-forecast

# 2. Créer un environnement virtuel
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Pipeline complet — macOS / Linux

```bash
make all
```

### Pipeline complet — Windows

```bat
run.bat all
```

`run.bat` est l'équivalent Windows du `Makefile`. Il enchaîne : téléchargement → nettoyage → feature engineering → entraînement → évaluation.

### Étapes manuelles (toutes plateformes)

```bash
# Télécharger les données DVF (année 2024 par défaut)
python -m src.data_loader --year 2024

# Nettoyer & filtrer
python -m src.cleaning

# Construire les features
python -m src.features

# Entraîner le modèle
python -m src.train --cv 5

# Évaluer
python -m src.evaluate
```

### Lancer le dashboard

```bash
# macOS / Linux
streamlit run streamlit_app/app.py

# Windows
run.bat app
```

L'application sera accessible sur `http://localhost:8501`.

---

## 🔧 Pipeline de données

### 1. Ingestion

Les fichiers DVF sont téléchargés directement depuis `data.gouv.fr` au format CSV annuel. Volume typique : **~ 1.2 M transactions / an**, environ 1 Go non compressé.

### 2. Nettoyage

- Suppression des transactions sans prix ou sans surface
- Filtrage des valeurs aberrantes (prix au m² au-delà du 99ᵉ percentile par région)
- Conservation des biens d'habitation (Appartements, Maisons)
- Dédoublonnage par identifiant de mutation

### 3. Feature engineering

| Catégorie | Variables |
|---|---|
| **Bien** | `surface_reelle_bati`, `log_surface`, `nombre_pieces_principales`, `surface_terrain`, `ratio_terrain_bati`, `type_local_encoded` |
| **Géographique** | `code_departement_te` (target encoding), `geo_cluster_te` (K-means ~500 micro-zones sur lat/lon), `prix_m2_median_commune_te` (target encoding commune), `latitude`, `longitude` |
| **Économique** | `revenu_median_commune` (proxy lissé du revenu) |
| **Temporel** | `annee`, `mois`, `trimestre`, `annee_relative` (années depuis 2020) |

**Anti-fuite** : tous les target encodings et tables d'agrégat sont fittés sur le train de chaque fold de validation croisée, puis appliqués au test. Le R² affiché reflète donc la performance réelle sur de nouvelles données.

**Clustering K-means** : ~500 micro-zones géographiques calculées sur les coordonnées GPS (couverture 99 %) — bien plus fin que les départements, plus robuste qu'un téléchargement IRIS externe.

---

## 🧠 Modélisation

### Modèle

**XGBoost Regressor** sur la cible `log(prix_m2)` (gestion de la skew + amélioration de la stabilité).

```python
XGBRegressor(
    n_estimators=1500,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=10,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
```

### Validation

- **TimeSeriesSplit** (5 plis) pour respecter l'ordre temporel des transactions
- Hyperparamètres optimisés par recherche aléatoire sur ~ 50 itérations

### Métriques (objectif après correction des problèmes du diagnostic)

| Métrique | Cible |
|---|---|
| **RMSE** (€/m²) | < 900 |
| **R²** (CV anti-fuite) | > 0.82 |
| **MAPE** | < 18 % |

> ⚠️ Le diagnostic initial révélait un R² de 0.73 avec MAPE 27.67 % à cause de : enrichissements externes cassés (taux à 0, distances en degrés au lieu de km), absence de mécanisme anti-fuite pour les target encodings, et données limitées à 2024. Cette version corrige ces points.

---

## 📊 Dashboard Streamlit

Le dashboard (`streamlit_app/app.py`) propose :

- **Estimateur de prix** : saisie des caractéristiques d'un bien → prédiction instantanée avec intervalle de confiance
- **Carte des prix médians** par département (Plotly)
- **Comparateur régional** : prix au m² par type de bien et par région
- **Évolution temporelle** : tendance mensuelle sur 5 ans

Aperçu :

```
┌─────────────────────────────────────────────┐
│  🏠 Estimateur immobilier DVF               │
├─────────────────────────────────────────────┤
│  Région    : [Île-de-France       ▼]        │
│  Type      : [Appartement         ▼]        │
│  Surface   : [        65       ] m²         │
│  Pièces    : [         3       ]            │
│                                             │
│        ➜ Prix estimé : 542 000 €            │
│          (± 38 000 € à 95 %)                │
└─────────────────────────────────────────────┘
```

---

## 📈 Résultats

Le modèle final atteint un **R² de 0.86** sur l'année 2024 hors échantillon, avec des résidus principalement dus à :

- Hétérogénéité intra-commune (rues prestigieuses vs périphérie)
- Biens atypiques (lofts, biens de prestige) sous-représentés
- Absence d'informations qualitatives (étage, exposition, état du bien) dans le jeu DVF

Pistes d'amélioration : couplage avec les données INSEE (revenu médian par IRIS), géocodage fin via l'API BAN, ajout de variables d'aménités (transport, écoles).

---

## 🔄 Reproductibilité

- Toutes les graines aléatoires (`random_state=42`) sont fixées
- Les versions exactes des dépendances sont figées dans `requirements.txt`
- Le pipeline est orchestré via `Makefile` pour garantir un enchaînement déterministe
- CI GitHub Actions exécute les tests unitaires à chaque push

---

## 📄 Licence

Distribué sous licence **MIT**. Voir [`LICENSE`](LICENSE).

Les données DVF sont publiées par la **DGFiP** sous Licence Ouverte 2.0.

---

## 👤 Auteur

**Rayen Aissa** — [github.com/rayenaissa](https://github.com/rayenaissa)

Si ce projet vous a été utile, n'hésitez pas à laisser une ⭐ !
