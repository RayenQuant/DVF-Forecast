"""Tests unitaires pour les modules features et enrich (version anti-fuite)."""

import numpy as np
import pandas as pd
import pytest

from src.enrich import (
    apply_target_encoding,
    build_commune_revenue_table,
    fit_target_encoding,
)
from src.features import (
    add_property_features,
    add_temporal_features,
    encode_type_local,
)


@pytest.fixture
def sample_clean_df():
    """Dataframe nettoyé minimal avec prix_m2 et log_prix_m2."""
    df = pd.DataFrame({
        "date_mutation": pd.to_datetime(
            ["2024-01-15", "2024-04-10", "2024-07-22", "2024-10-05"] * 5
        ),
        "type_local": ["Appartement", "Maison"] * 10,
        "code_departement": ["75", "31", "75", "13"] * 5,
        "code_commune": ["75101", "31555", "75112", "13201"] * 5,
        "valeur_fonciere": [400_000, 350_000, 600_000, 250_000] * 5,
        "surface_reelle_bati": [50, 100, 70, 80] * 5,
        "surface_terrain": [0, 500, 0, 300] * 5,
        "nombre_pieces_principales": [2, 4, 3, 3] * 5,
        "prix_m2": [8000.0, 3500.0, 8571.0, 3125.0] * 5,
        "latitude": [48.85, 43.60, 48.85, 43.30] * 5,
        "longitude": [2.35, 1.44, 2.35, 5.37] * 5,
    })
    df["log_prix_m2"] = np.log1p(df["prix_m2"])
    return df


def test_temporal_features(sample_clean_df):
    out = add_temporal_features(sample_clean_df)
    assert "annee" in out.columns
    assert "mois" in out.columns
    assert "trimestre" in out.columns
    assert "annee_relative" in out.columns
    assert out["annee"].iloc[0] == 2024
    assert out["annee_relative"].iloc[0] == 4   # 2024 - 2020


def test_property_features_ratio(sample_clean_df):
    out = add_property_features(sample_clean_df)
    assert "ratio_terrain_bati" in out.columns
    # Maison ligne 1 : 500/100 = 5.0
    assert out["ratio_terrain_bati"].iloc[1] == pytest.approx(5.0)
    assert "log_surface" in out.columns
    assert (out["log_surface"] > 0).all()


def test_property_features_no_inf(sample_clean_df):
    df = sample_clean_df.copy()
    df.loc[0, "surface_reelle_bati"] = 0
    out = add_property_features(df)
    assert np.isfinite(out["ratio_terrain_bati"]).all()


def test_encode_type_local_fit(sample_clean_df):
    out, encoder = encode_type_local(sample_clean_df)
    assert "type_local_encoded" in out.columns
    assert pd.api.types.is_integer_dtype(out["type_local_encoded"])
    assert encoder.classes_.tolist() == ["Appartement", "Maison"]


def test_encode_type_local_apply(sample_clean_df):
    _, encoder = encode_type_local(sample_clean_df)
    new_df = sample_clean_df.head(2).copy()
    new_df.loc[0, "type_local"] = "Inconnu"
    out, _ = encode_type_local(new_df, encoder=encoder)
    assert "type_local_encoded" in out.columns


def test_target_encoding_smoothing(sample_clean_df):
    """Vérifie que le smoothing rapproche bien les petites catégories
    de la moyenne globale."""
    encoding = fit_target_encoding(sample_clean_df, "code_departement",
                                    target="log_prix_m2", smoothing=10.0)
    global_mean = encoding["__global__"]
    assert "75" in encoding
    assert "__global__" in encoding
    # Avec smoothing fort, les valeurs encodées doivent être proches de la moyenne globale
    assert abs(encoding["75"] - global_mean) < 1.0


def test_target_encoding_unknown_category(sample_clean_df):
    """Vérifie que les catégories inconnues reçoivent la moyenne globale."""
    encoding = fit_target_encoding(sample_clean_df, "code_departement",
                                    target="log_prix_m2", smoothing=10.0)
    new_df = pd.DataFrame({"code_departement": ["99"]})  # dept inconnu
    out = apply_target_encoding(new_df, "code_departement", encoding)
    assert out["code_departement_te"].iloc[0] == encoding["__global__"]


def test_commune_revenue_table(sample_clean_df):
    table = build_commune_revenue_table(sample_clean_df)
    assert "code_commune" in table.columns
    assert "revenu_median_commune" in table.columns
    assert (table["revenu_median_commune"] > 0).all()
    # Borne supérieure : 80k
    assert (table["revenu_median_commune"] <= 80_000).all()


def test_anti_leakage_fit_only_on_train():
    """Test critique : fit_target_encoding utilise SEULEMENT le train."""
    train_df = pd.DataFrame({
        "code_departement": ["A", "A", "B", "B"],
        "log_prix_m2": [10.0, 10.5, 8.0, 8.5],
    })
    test_df = pd.DataFrame({
        "code_departement": ["A", "B"],
        "log_prix_m2": [50.0, 50.0],  # Si fuite, ces valeurs aberrantes
                                       # affecteraient l'encoding
    })
    encoding = fit_target_encoding(train_df, "code_departement",
                                    target="log_prix_m2", smoothing=5.0)
    out = apply_target_encoding(test_df, "code_departement", encoding)
    # Les valeurs encodées doivent refléter le train, pas le test
    assert out["code_departement_te"].iloc[0] < 15.0  # < valeurs aberrantes du test
