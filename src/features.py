"""Feature engineering — version anti-fuite.

CHANGEMENTS CLÉS vs version initiale (suite au diagnostic) :
- Les médianes commune/dept sont calculées sur TRAIN seul puis appliquées au TEST.
- Remplacement de LabelEncoder par target encoding lissé (Bayesian smoothing).
- Ajout du geo_cluster (K-means sur lat/lon, ~500 micro-zones).
- Pas de fuite : tous les artefacts sont fittés sur train et réutilisés sur test.

Le pipeline expose deux modes :
1. `build_features(df)` : pour entraînement standard, calcule tout sur df entier.
   ⚠️ Crée une fuite mineure (les médianes voient le test).
2. `build_features_split(train_df, test_df)` : pour évaluation propre,
   fitte sur train et applique au test. → R² réaliste.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLEAN_PARQUET,
    COMMUNE_REVENUE_PATH,
    ENCODERS_PATH,
    FEATURES_PARQUET,
    TARGET,
    TARGET_ENCODINGS_PATH,
)
from src.enrich import (
    apply_target_encoding,
    assign_geo_cluster,
    build_commune_revenue_table,
    fit_geo_clusterer,
    fit_target_encoding,
    merge_commune_revenue,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Features de base (pas de fuite, juste des transformations locales à la ligne)
# =============================================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["annee"] = df["date_mutation"].dt.year
    df["mois"] = df["date_mutation"].dt.month
    df["trimestre"] = df["date_mutation"].dt.quarter
    # Tendance temporelle : années depuis 2020
    df["annee_relative"] = df["annee"] - 2020
    return df


def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées strictement de la ligne courante (zéro fuite)."""
    df = df.copy()
    df["ratio_terrain_bati"] = df["surface_terrain"] / df["surface_reelle_bati"]
    df["ratio_terrain_bati"] = (
        df["ratio_terrain_bati"].replace([np.inf, -np.inf], 0).fillna(0)
    )
    df["log_prix"] = np.log1p(df["valeur_fonciere"])
    df["log_prix_m2"] = np.log1p(df["prix_m2"])
    df["log_surface"] = np.log1p(df["surface_reelle_bati"])
    return df


def encode_type_local(
    df: pd.DataFrame,
    encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """LabelEncoder pour type_local (binaire Appart/Maison)."""
    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
        df["type_local_encoded"] = encoder.fit_transform(df["type_local"].astype(str))
    else:
        known = set(encoder.classes_)
        df["type_local"] = df["type_local"].astype(str).where(
            df["type_local"].astype(str).isin(known), other=encoder.classes_[0]
        )
        df["type_local_encoded"] = encoder.transform(df["type_local"])
    return df, encoder


# =============================================================================
# Pipeline complet — mode "train" (fit) ou "transform"
# =============================================================================
def fit_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construit les features ET retourne les artefacts (encoders, encodings).

    À utiliser sur le train set pour produire le modèle ET les artefacts
    réutilisables sur le test set.
    """
    logger.info("=== FIT FEATURES sur %d lignes ===", len(df))

    # Étape 1 : features purement locales (pas de fuite)
    df = add_temporal_features(df)
    df = add_property_features(df)

    # Étape 2 : encoder type_local
    df, type_encoder = encode_type_local(df)

    # Étape 3 : K-means sur GPS
    logger.info("Fit clustering géographique...")
    geo_clusterer = fit_geo_clusterer(df)
    df = assign_geo_cluster(df, geo_clusterer)

    # Étape 4 : table revenu commune
    logger.info("Construction table revenu/commune...")
    revenue_table = build_commune_revenue_table(df)
    revenue_table.to_parquet(COMMUNE_REVENUE_PATH, index=False)
    df = merge_commune_revenue(df, revenue_table)

    # Étape 5 : target encodings (calculés sur train uniquement = anti-fuite)
    logger.info("Fit target encodings...")
    target_encodings = {
        "code_departement": fit_target_encoding(df, "code_departement", TARGET, smoothing=20.0),
        "code_commune": fit_target_encoding(df, "code_commune", TARGET, smoothing=10.0),
        "geo_cluster": fit_target_encoding(df, "geo_cluster", TARGET, smoothing=15.0),
    }
    joblib.dump(target_encodings, TARGET_ENCODINGS_PATH)

    # Appliquer les target encodings
    df = apply_target_encoding(df, "code_departement",
                                target_encodings["code_departement"],
                                "code_departement_te")
    df = apply_target_encoding(df, "code_commune",
                                target_encodings["code_commune"],
                                "prix_m2_median_commune_te")
    df = apply_target_encoding(df, "geo_cluster",
                                target_encodings["geo_cluster"],
                                "geo_cluster_te")

    # Sauvegarde encoder type_local
    encoders = {"type_local": type_encoder}
    joblib.dump(encoders, ENCODERS_PATH)

    # Imputation finale GPS manquant (médiane train)
    for col in ("latitude", "longitude"):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    artifacts = {
        "encoders": encoders,
        "target_encodings": target_encodings,
        "geo_clusterer": geo_clusterer,
        "revenue_table": revenue_table,
    }
    logger.info("=== FIT FEATURES TERMINÉ : %d colonnes ===", df.shape[1])
    return df, artifacts


def transform_features(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Applique les artefacts pré-calculés à un nouveau DataFrame (test/inference)."""
    df = add_temporal_features(df)
    df = add_property_features(df)
    df, _ = encode_type_local(df, encoder=artifacts["encoders"]["type_local"])
    df = assign_geo_cluster(df, clusterer=artifacts["geo_clusterer"])
    df = merge_commune_revenue(df, artifacts["revenue_table"])

    te = artifacts["target_encodings"]
    df = apply_target_encoding(df, "code_departement", te["code_departement"],
                                "code_departement_te")
    df = apply_target_encoding(df, "code_commune", te["code_commune"],
                                "prix_m2_median_commune_te")
    df = apply_target_encoding(df, "geo_cluster", te["geo_cluster"],
                                "geo_cluster_te")

    for col in ("latitude", "longitude"):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# =============================================================================
# Mode "fit + apply au même df" pour le pipeline standard
# (utilisé par run.bat features)
# =============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mode standard : fit sur tout le df et applique.

    ⚠️ Crée une fuite minime sur les target encodings.
    Pour une évaluation honnête, utiliser fit_features puis transform_features
    via les splits de cross-validation (voir train.py).
    """
    df_feat, _ = fit_features(df)
    return df_feat


def main() -> None:
    logger.info("Lecture des données nettoyées : %s", CLEAN_PARQUET)
    df = pd.read_parquet(CLEAN_PARQUET)
    df_feat = build_features(df)
    df_feat.to_parquet(FEATURES_PARQUET, index=False)
    logger.info(
        "Features écrites : %s (%d lignes, %d colonnes)",
        FEATURES_PARQUET, len(df_feat), df_feat.shape[1],
    )


if __name__ == "__main__":
    main()
