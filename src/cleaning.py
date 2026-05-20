"""Nettoyage des données DVF brutes — version renforcée.

Améliorations vs version initiale (suite au diagnostic) :
- Filtre prix/m² plus strict : [500 ; 20 000]
- Cap pieces <= 15 (max actuel : 198 = aberrant)
- Cap ratio terrain/bâti <= 100 (max actuel : 11 536 = aberrant)
- Winsorisation par dept renforcée pour les départements à forte hétérogénéité (CV > 65%)
- Filtre GPS : exclut les transactions hors France métropolitaine + DOM principaux
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_PARQUET,
    FRANCE_METRO_BBOX,
    HIGH_HET_CV_THRESHOLD,
    MAX_PIECES,
    MAX_PRIX_EUR,
    MAX_PRIX_M2,
    MAX_RATIO_TERRAIN_BATI,
    MAX_SURFACE_M2,
    MIN_PRIX_EUR,
    MIN_PRIX_M2,
    MIN_SURFACE_M2,
    NATURES_MUTATION,
    RAW_PARQUET,
    TYPES_LOCAUX,
    WINSOR_HIGH_HET_LOWER_Q,
    WINSOR_HIGH_HET_UPPER_Q,
    WINSOR_LOWER_Q,
    WINSOR_UPPER_Q,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def log_step(name: str, before: int, after: int) -> None:
    pct = 100 * (before - after) / before if before > 0 else 0
    logger.info(
        "%-55s : %8d -> %8d  (-%d, -%.1f%%)",
        name, before, after, before - after, pct,
    )


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    df["nombre_pieces_principales"] = pd.to_numeric(
        df["nombre_pieces_principales"], errors="coerce"
    ).fillna(0)
    df["surface_terrain"] = pd.to_numeric(df["surface_terrain"], errors="coerce").fillna(0)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    return df.dropna(subset=["date_mutation", "valeur_fonciere"])


def filter_nature_and_type(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["nature_mutation"].isin(NATURES_MUTATION)].copy()
    log_step("Filtre nature = Vente", n0, len(df))

    n0 = len(df)
    df = df[df["type_local"].isin(TYPES_LOCAUX)].copy()
    log_step("Filtre type (Appart./Maison)", n0, len(df))
    return df


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    essentials = [
        "valeur_fonciere", "surface_reelle_bati", "code_departement",
        "date_mutation", "type_local",
    ]
    df = df.dropna(subset=essentials).copy()
    log_step("Suppression NaN essentiels", n0, len(df))
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filtres aberrants renforcés."""
    n0 = len(df)
    df = df[(df["surface_reelle_bati"] >= MIN_SURFACE_M2) &
            (df["surface_reelle_bati"] <= MAX_SURFACE_M2)].copy()
    log_step(f"Surface dans [{MIN_SURFACE_M2}, {MAX_SURFACE_M2}] m²", n0, len(df))

    n0 = len(df)
    df = df[(df["valeur_fonciere"] >= MIN_PRIX_EUR) &
            (df["valeur_fonciere"] <= MAX_PRIX_EUR)].copy()
    log_step(f"Prix dans [{MIN_PRIX_EUR:,}, {MAX_PRIX_EUR:,}] €", n0, len(df))

    # Cap pieces (problème #6 : max 198 dans le diag)
    n0 = len(df)
    df = df[df["nombre_pieces_principales"] <= MAX_PIECES].copy()
    log_step(f"Nb pieces <= {MAX_PIECES}", n0, len(df))

    # Prix au m²
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    n0 = len(df)
    df = df[(df["prix_m2"] >= MIN_PRIX_M2) &
            (df["prix_m2"] <= MAX_PRIX_M2)].copy()
    log_step(f"Prix/m² dans [{MIN_PRIX_M2}, {MAX_PRIX_M2:,}] €", n0, len(df))

    # Cap ratio terrain/bâti (problème #6 : max 11 536 dans le diag)
    n0 = len(df)
    ratio = df["surface_terrain"] / df["surface_reelle_bati"].replace(0, np.nan)
    df = df[(ratio.isna()) | (ratio <= MAX_RATIO_TERRAIN_BATI)].copy()
    log_step(f"Ratio terrain/bâti <= {MAX_RATIO_TERRAIN_BATI}", n0, len(df))

    return df


def filter_gps_france(df: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les coordonnées dans la France métro + DOM proches.

    On accepte les lignes sans GPS (on les remplira plus tard).
    On rejette uniquement les GPS clairement aberrants.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    n0 = len(df)
    no_gps = df["latitude"].isna() | df["longitude"].isna()
    gps_ok = (
        df["latitude"].between(FRANCE_METRO_BBOX["lat_min"],
                                FRANCE_METRO_BBOX["lat_max"]) &
        df["longitude"].between(FRANCE_METRO_BBOX["lon_min"],
                                 FRANCE_METRO_BBOX["lon_max"])
    )
    # Inclut DOM-TOM par leurs vrais bbox
    dom = (
        df["latitude"].between(-22, 16) &
        df["longitude"].between(-63, 56)
    )
    keep = no_gps | gps_ok | dom
    df = df[keep].copy()
    log_step("Filtre GPS valide (ou absent)", n0, len(df))
    return df


def winsorize_by_dept(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorisation adaptative par département.

    - Départements à forte hétérogénéité (CV > 65%) : winsorisation 2.5%/97.5%
    - Autres : winsorisation 1%/99%
    """
    if "prix_m2" not in df.columns:
        return df

    n0 = len(df)

    # Calcul du CV par dept
    by_dept = df.groupby("code_departement")["prix_m2"].agg(["mean", "std", "count"])
    by_dept["cv"] = (by_dept["std"] / by_dept["mean"] * 100).fillna(0)
    high_het_depts = set(by_dept[by_dept["cv"] > HIGH_HET_CV_THRESHOLD].index.tolist())
    logger.info(
        "Départements à forte hétérogénéité (CV > %.0f%%) : %d / %d",
        HIGH_HET_CV_THRESHOLD, len(high_het_depts), len(by_dept),
    )

    def trim(group: pd.DataFrame) -> pd.DataFrame:
        dept = group.name
        if dept in high_het_depts:
            lo_q, hi_q = WINSOR_HIGH_HET_LOWER_Q, WINSOR_HIGH_HET_UPPER_Q
        else:
            lo_q, hi_q = WINSOR_LOWER_Q, WINSOR_UPPER_Q
        q_low = group["prix_m2"].quantile(lo_q)
        q_high = group["prix_m2"].quantile(hi_q)
        return group[(group["prix_m2"] >= q_low) & (group["prix_m2"] <= q_high)]

    df = df.groupby("code_departement", group_keys=False).apply(trim)
    log_step("Winsorisation adaptative par département", n0, len(df))
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if "id_mutation" not in df.columns:
        return df
    n0 = len(df)
    df = df.drop_duplicates(subset=["id_mutation"]).copy()
    log_step("Dédoublonnage id_mutation", n0, len(df))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== DÉMARRAGE NETTOYAGE : %d lignes brutes ===", len(df))
    df = cast_types(df)
    df = filter_nature_and_type(df)
    df = drop_missing(df)
    df = filter_outliers(df)
    df = filter_gps_france(df)
    df = winsorize_by_dept(df)
    df = dedupe(df)
    logger.info("=== NETTOYAGE TERMINÉ : %d lignes finales ===", len(df))
    return df


def main() -> None:
    logger.info("Lecture du fichier brut : %s", RAW_PARQUET)
    df = pd.read_parquet(RAW_PARQUET)
    df_clean = clean(df)
    df_clean.to_parquet(CLEAN_PARQUET, index=False)
    logger.info("Fichier nettoyé écrit : %s", CLEAN_PARQUET)


if __name__ == "__main__":
    main()
