"""Interface de prédiction pour un bien individuel.

Charge le modèle + artefacts et applique exactement le même pipeline
de features (transform_features) qu'au moment de l'entraînement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import joblib
import numpy as np
import pandas as pd

from src.config import (
    COMMUNE_REVENUE_PATH,
    ENCODERS_PATH,
    FEATURE_COLUMNS,
    GEO_CLUSTERER_PATH,
    MODEL_PATH,
    TARGET_ENCODINGS_PATH,
)
from src.features import transform_features

logger = logging.getLogger(__name__)


@dataclass
class BienImmobilier:
    """Représentation d'un bien à estimer."""
    type_local: str                      # "Appartement" ou "Maison"
    surface_reelle_bati: float
    nombre_pieces_principales: int
    code_departement: str
    code_commune: str | None = None
    surface_terrain: float = 0.0
    latitude: float | None = None
    longitude: float | None = None
    date_mutation: date | None = None


class PricePredictor:
    """Wrapper de prédiction utilisant les mêmes artefacts que l'entraînement."""

    def __init__(self):
        # Chargement modèle + tous les artefacts
        self.model = joblib.load(MODEL_PATH)
        self.artifacts = {
            "encoders": joblib.load(ENCODERS_PATH),
            "target_encodings": joblib.load(TARGET_ENCODINGS_PATH),
            "geo_clusterer": joblib.load(GEO_CLUSTERER_PATH),
            "revenue_table": pd.read_parquet(COMMUNE_REVENUE_PATH),
        }

        # Médianes lat/lon par département (fallback si GPS pas fourni)
        rev_table = self.artifacts["revenue_table"]
        self.dept_centroids: dict[str, tuple[float, float]] = {}
        # Note: pas de centroids stockés directement, on utilisera les coords médianes
        # via une approche basée sur le département dans les communes connues.

    def predict(self, bien: BienImmobilier) -> dict:
        d = bien.date_mutation or date.today()

        # Si lat/lon non fournis, on utilisera les médianes plus tard
        lat = bien.latitude if bien.latitude is not None else 46.5  # centre France
        lon = bien.longitude if bien.longitude is not None else 2.5

        # Construire le DataFrame d'entrée au même format que le pipeline
        row = pd.DataFrame([{
            "date_mutation": pd.Timestamp(d),
            "type_local": bien.type_local,
            "surface_reelle_bati": float(bien.surface_reelle_bati),
            "nombre_pieces_principales": int(bien.nombre_pieces_principales),
            "surface_terrain": float(bien.surface_terrain),
            "code_departement": str(bien.code_departement),
            "code_commune": bien.code_commune or f"{bien.code_departement}000",
            "latitude": float(lat),
            "longitude": float(lon),
            "valeur_fonciere": 0.0,    # placeholder, pas utilisé pour prédire
            "prix_m2": 0.0,             # placeholder
        }])

        # Appliquer le pipeline complet
        row_feat = transform_features(row, self.artifacts)

        # Vérifier que toutes les features sont là
        missing = [c for c in FEATURE_COLUMNS if c not in row_feat.columns]
        if missing:
            raise RuntimeError(f"Features manquantes : {missing}")

        X = row_feat[FEATURE_COLUMNS]
        log_prix_m2 = float(self.model.predict(X)[0])
        prix_m2 = float(np.expm1(log_prix_m2))
        prix_total = prix_m2 * bien.surface_reelle_bati

        return {
            "prix_m2_estime": round(prix_m2, 0),
            "prix_total_estime": round(prix_total, 0),
            "revenu_median_commune_estime": round(
                float(row_feat["revenu_median_commune"].iloc[0]), 0
            ),
            "geo_cluster": int(row_feat["geo_cluster"].iloc[0]),
            "code_departement_te": round(
                float(row_feat["code_departement_te"].iloc[0]), 3
            ),
        }


def demo() -> None:
    """Démo CLI rapide."""
    bien = BienImmobilier(
        type_local="Appartement",
        surface_reelle_bati=65,
        nombre_pieces_principales=3,
        code_departement="75",
        code_commune="75112",
        latitude=48.8403,
        longitude=2.3956,
    )
    predictor = PricePredictor()
    result = predictor.predict(bien)
    print("\n=== Prédiction pour appartement Paris 12e (65 m², T3) ===")
    for k, v in result.items():
        print(f"  {k:35s} : {v:>14,.0f}".replace(",", "\u202f"))


if __name__ == "__main__":
    demo()
