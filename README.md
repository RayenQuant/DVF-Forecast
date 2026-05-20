#  Prévision de Prix Immobiliers — DVF Open Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/rayenaissa/dvf-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/rayenaissa/dvf-forecast/actions)

Modélisation et prévision des prix de l'immobilier en France à partir des **Demandes de Valeurs Foncières (DVF)** publiées en open data sur [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/). Le pipeline complet ingère et nettoie **plus de 3,5 millions de transactions brutes (2020–2024)**, construit des variables géographiques, typologiques et temporelles, entraîne un modèle **XGBoost** avec validation croisée temporelle anti-fuite, et expose les prédictions via un **dashboard Streamlit** interactif.

---

##  Sommaire

- [Aperçu](#-aperçu)
- [Résultats](#-résultats)
- [Architecture du projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline de données](#-pipeline-de-données)
- [Modélisation](#-modélisation)
- [Dashboard Streamlit](#-dashboard-streamlit)
- [Reproductibilité](#-reproductibilité)
- [Licence](#-licence)

---

##  Aperçu

Ce projet illustre un **workflow data science de bout en bout** appliqué au marché immobilier français :

- **Ingestion multi-années** : téléchargement automatique des fichiers DVF 2020–2024 (~3,5 M transactions brutes)
- **Nettoyage robuste** : filtres aberrants, winsorisation adaptative par département (stricte pour les zones à forte hétérogénéité), caps surface/pièces/ratio terrain, dédoublonnage
- **Feature engineering anti-fuite** : target encoding lissé (Bayesian smoothing) fitté uniquement sur le train, clustering K-means géographique (~500 micro-zones sur lat/lon)
- **Modèle XGBoost** avec TimeSeriesSplit (5 folds) — le test ne voit jamais les encodings du train
- **Dashboard Streamlit** permettant d'estimer le prix d'un bien en saisissant ses caractéristiques et coordonnées GPS

### Stack technique

| Domaine | Outils |
|---|---|
| Ingestion & nettoyage | Python, pandas, pyarrow |
| Feature engineering | scikit-learn (KMeans, LabelEncoder) |
| Modélisation | XGBoost, scikit-learn |
| Visualisation | Plotly, matplotlib, seaborn |
| Déploiement | Streamlit |
| Qualité du code | pytest (17 tests), ruff, black |
| CI/CD | GitHub Actions |

---

##  Résultats

Métriques obtenues en **validation croisée temporelle anti-fuite** sur 5 années (2020–2024), ~730k transactions après nettoyage :

| Métrique | Valeur |
|---|---|
| **R²** | **0.753** |
| **RMSE** | **1 198 €/m²** |
| **MAPE** | **25.73 %** |

La variance des folds est faible (±0.007 sur R²), ce qui indique un modèle stable dans le temps.

### Pourquoi les estimations sont inférieures aux prix affichés en agence

Le DVF enregistre les **prix de transaction** (ce qui a été réellement payé chez le notaire). Les sites immobiliers affichent des **prix de demande** (ce que le vendeur espère obtenir). L'écart structurel est de :

- **Paris / grandes villes** : +8 à +15 %
- **Villes moyennes** : +3 à +8 %

De plus, le DVF ne contient pas l'étage, l'état du bien, l'exposition, ni la présence d'un parking — des facteurs qui représentent ±10–30 % de variation supplémentaire. Le plafond réaliste d'un modèle basé sur DVF seul est estimé à **R² ≈ 0.80**.

---

##  Architecture du projet

```
dvf-forecast/
├── data/
│   ├── raw/                  # Fichiers DVF bruts (CSV depuis data.gouv.fr) — non versionné
│   └── processed/            # Données nettoyées & features (parquet) — non versionné
├── src/
│   ├── __init__.py
│   ├── config.py             # Constantes, chemins, hyperparamètres
│   ├── data_loader.py        # Téléchargement multi-années (2020–2024 par défaut)
│   ├── cleaning.py           # Nettoyage + winsorisation adaptative par département
│   ├── enrich.py             # K-means géographique + target encoding anti-fuite
│   ├── features.py           # Feature engineering (fit/transform strictement séparés)
│   ├── train.py              # XGBoost + TimeSeriesSplit anti-fuite
│   ├── evaluate.py           # Métriques RMSE / R² / MAPE par segment
│   └── predict.py            # Interface de prédiction (bien individuel)
├── notebooks/
│   ├── 01_exploration.ipynb  # EDA initiale
│   ├── 02_features.ipynb     # Conception des variables
│   ├── 03_modeling.ipynb     # Itérations modèles
│   └── 04_diagnostic.ipynb   # Diagnostic complet (14 sections, 730k lignes)
├── streamlit_app/
│   └── app.py                # Dashboard interactif
├── models/                   # Modèles + artefacts (.joblib) — non versionné
├── tests/                    # 17 tests unitaires (pytest)
├── sql/queries.sql           # Requêtes d'agrégation exploratoires
├── requirements.txt          # Dépendances Python (compatible 3.10–3.13)
├── Makefile                  # Orchestration Unix/macOS
├── run.bat                   # Orchestration Windows
├── .github/workflows/ci.yml  # CI : lint + tests sur Python 3.10 et 3.11
└── README.md
```

---

## ⚙️ Installation

### Prérequis

- Python **3.10+** (testé jusqu'à 3.13) — Windows, macOS, Linux
- ~8 Go d'espace disque (5 fichiers DVF + données traitées)
- ~4 Go de RAM pour le traitement

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

##  Utilisation

### Pipeline complet

```bash
# macOS / Linux
make all

# Windows
run.bat all
```

Enchaîne automatiquement : téléchargement 2020–2024 → nettoyage → feature engineering → entraînement → évaluation. Durée : **~20–30 minutes** (dont ~10 min de téléchargement).

### Commandes individuelles

```bash
# Windows                          # macOS / Linux
run.bat download                   make download        # Télécharger 2020–2024
run.bat download-1y                make download-1y     # Télécharger 2024 seulement (test rapide)
run.bat clean                      make clean           # Nettoyage
run.bat features                   make features        # Feature engineering
run.bat train                      make train           # Entraînement XGBoost
run.bat evaluate                   make evaluate        # Évaluation hold-out
run.bat app                        make app             # Dashboard Streamlit
run.bat test                       make test            # Tests unitaires
```

### Dashboard Streamlit

```bash
run.bat app   # Windows
make app      # macOS / Linux
```

Accessible sur `http://localhost:8501`.

---

## 🔧 Pipeline de données

### 1. Ingestion

Téléchargement en streaming des fichiers DVF annuels depuis `data.gouv.fr` (~90 Mo compressés par année). Lecture par chunks de 200k lignes pour limiter l'empreinte mémoire.

### 2. Nettoyage

| Étape | Détail |
|---|---|
| Filtres métier | Ventes uniquement, Appartements et Maisons uniquement |
| Valeurs manquantes | Suppression sur prix, surface, département, date |
| Aberrants surface | [9 ; 800] m² |
| Aberrants prix | [10k ; 10M] € |
| Aberrants pièces | ≤ 15 (le max observé dans le brut était 198 — erreur de saisie) |
| Aberrants prix/m² | [500 ; 20 000] €/m² |
| Ratio terrain/bâti | ≤ 100 (le max observé était 11 536 — terrain agricole mal classé) |
| GPS | Rejet des coordonnées hors France |
| Winsorisation | 1%/99% standard — 2.5%/97.5% pour les 82 départements à forte hétérogénéité (CV > 65%) |
| Dédoublonnage | Par `id_mutation` |

**Résultat** : ~730k transactions propres sur 5 années (~79% des lignes Appartements/Maisons).

### 3. Feature engineering

| Catégorie | Variables |
|---|---|
| **Bien** | `surface_reelle_bati`, `log_surface`, `nombre_pieces_principales`, `surface_terrain`, `ratio_terrain_bati`, `type_local_encoded` |
| **Géographique** | `latitude`, `longitude`, `code_departement_te`, `geo_cluster_te` (K-means 500 micro-zones), `prix_m2_median_commune_te` |
| **Économique** | `revenu_median_commune` (proxy lissé calculé depuis les données DVF) |
| **Temporel** | `annee`, `mois`, `trimestre`, `annee_relative` (années depuis 2020) |

**Anti-fuite** : les target encodings et tables d'agrégat sont calculés uniquement sur le fold de train, puis appliqués au fold de test. Le R² mesuré est donc une estimation honnête de la performance sur données inconnues.

**K-means géographique** : 500 micro-zones calculées sur les coordonnées GPS (couverture 99.2 %) — plus fin que le département, plus robuste qu'un téléchargement IRIS externe potentiellement instable.

---

## 🧠 Modélisation

### Modèle

**XGBoost Regressor** sur la cible `log(prix_m²)` (stabilise la distribution asymétrique, skewness passant de ~3.2 à ~0.03 après transformation).

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

### Validation croisée anti-fuite

```
Données triées chronologiquement
│
├─ Fold 1 : train [jan 2020 – mar 2022] | test [mar – jun 2022]
├─ Fold 2 : train [jan 2020 – jun 2022] | test [jun – sep 2022]
├─ Fold 3 : train [jan 2020 – sep 2022] | test [sep – dec 2022]
├─ Fold 4 : train [jan 2020 – dec 2022] | test [dec 2022 – mar 2023]
└─ Fold 5 : train [jan 2020 – mar 2023] | test [mar – jun 2023]

Pour chaque fold :
  1. fit_features(train)  → calcule encodings, clusters, médianes
  2. transform(test)      → applique les artefacts du train
  3. XGBoost.fit(train) + early_stopping sur val interne
  4. Évaluation sur test → RMSE, R², MAPE
```

---

##  Dashboard Streamlit

4 pages :

- ** Estimateur** : saisie des caractéristiques + coordonnées GPS → prix estimé au m² et total
- ** Exploration** : distribution des prix, top départements, évolution trimestrielle 2020–2024
- ** Performance** : métriques CV par fold, graphiques RMSE et R²
- ** À propos** : documentation du pipeline et des sources

---

##  Reproductibilité

- `random_state=42` fixé partout (KMeans, XGBoost, splits)
- Dépendances Python avec borne minimale dans `requirements.txt` (compatible 3.10–3.13)
- Pipeline orchestré via `Makefile` / `run.bat` pour un enchaînement déterministe
- CI GitHub Actions : lint (ruff, black) + 17 tests unitaires à chaque push

---

## 📄 Licence

Distribué sous licence **MIT**. Voir [`LICENSE`](LICENSE).

Les données DVF sont publiées par la **DGFiP** sous Licence Ouverte 2.0.

---

## 👤 Auteur

**Rayen Aissa** 

Si ce projet vous a été utile, n'hésitez pas à laisser une ⭐ !
