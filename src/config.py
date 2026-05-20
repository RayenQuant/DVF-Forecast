"""Configuration globale du projet : chemins, constantes, paramètres."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
SQL_DIR = ROOT_DIR / "sql"

for _p in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Données DVF
# ---------------------------------------------------------------------------
DVF_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"

# Années par défaut (multi-année = capture les cycles de marché)
DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024]

RAW_COLUMNS = [
    "id_mutation",
    "date_mutation",
    "nature_mutation",
    "valeur_fonciere",
    "adresse_nom_voie",
    "code_postal",
    "code_commune",
    "nom_commune",
    "code_departement",
    "type_local",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "longitude",
    "latitude",
]

TYPES_LOCAUX = ["Appartement", "Maison"]
NATURES_MUTATION = ["Vente"]

# ---------------------------------------------------------------------------
# Filtres anti-aberrants
# ---------------------------------------------------------------------------
MIN_SURFACE_M2 = 9
MAX_SURFACE_M2 = 800
MIN_PRIX_EUR = 10_000
MAX_PRIX_EUR = 10_000_000
MIN_PRIX_M2 = 500            # +strict : élimine biens à rénover en zone rurale
MAX_PRIX_M2 = 20_000         # +strict : Paris 6/7 peut dépasser mais rare

# Caps absolus supplémentaires (problème #6 du diagnostic)
MAX_PIECES = 15
MAX_RATIO_TERRAIN_BATI = 100.0   # un terrain > 100x le bâti = aberration

# Winsorisation par département (déjà efficace)
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99
# Winsorisation +stricte pour départements à forte hétérogénéité
WINSOR_HIGH_HET_LOWER_Q = 0.025
WINSOR_HIGH_HET_UPPER_Q = 0.975
HIGH_HET_CV_THRESHOLD = 65.0     # CV > 65% -> winsorisation +stricte

# ---------------------------------------------------------------------------
# Géocodage / clustering géographique
# ---------------------------------------------------------------------------
# Bornes France métropolitaine + DOM-TOM principaux
FRANCE_METRO_BBOX = {
    "lat_min": 41.0, "lat_max": 51.5,
    "lon_min": -5.5, "lon_max": 10.0,
}
# Nombre de micro-zones K-means (lat/lon) — ~500 fournit ~3km de résolution
N_GEO_CLUSTERS = 500

# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
TARGET = "log_prix_m2"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    # ── Caractéristiques du bien ──
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "type_local_encoded",
    "ratio_terrain_bati",
    "log_surface",                       # capture non-linéarité surface
    # ── Localisation ──
    "code_departement_te",               # target encoding (vs label encoding)
    "geo_cluster_te",                    # target encoding du cluster K-means
    "prix_m2_median_commune_te",         # target encoding commune (anti-fuite)
    # ── Coordonnées brutes (laissées au modèle) ──
    "latitude",
    "longitude",
    # ── Enrichissement revenu (commune-level, fiable) ──
    "revenu_median_commune",
    # ── Temporel ──
    "annee",
    "mois",
    "trimestre",
    "annee_relative",                    # années depuis 2020 (capture la tendance)
]

XGB_PARAMS = {
    "n_estimators": 1500,
    "max_depth": 10,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

# ---------------------------------------------------------------------------
# Fichiers produits par le pipeline
# ---------------------------------------------------------------------------
RAW_PARQUET = PROCESSED_DIR / "dvf_raw.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "dvf_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "dvf_features.parquet"
MODEL_PATH = MODELS_DIR / "xgb_dvf.joblib"
ENCODERS_PATH = MODELS_DIR / "encoders.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Artefacts target encoding & clustering
TARGET_ENCODINGS_PATH = MODELS_DIR / "target_encodings.joblib"
GEO_CLUSTERER_PATH = MODELS_DIR / "geo_clusterer.joblib"
COMMUNE_REVENUE_PATH = PROCESSED_DIR / "commune_revenue.parquet"
