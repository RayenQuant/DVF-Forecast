"""Tests unitaires pour le module de nettoyage."""

import pandas as pd
import pytest

from src.cleaning import (
    cast_types,
    dedupe,
    drop_missing,
    filter_gps_france,
    filter_nature_and_type,
    filter_outliers,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id_mutation": ["m1", "m2", "m2", "m3", "m4"],
        "date_mutation": ["2024-01-15", "2024-02-20", "2024-02-20",
                          "2024-03-10", "2024-04-01"],
        "nature_mutation": ["Vente", "Vente", "Vente", "Echange", "Vente"],
        "valeur_fonciere": [250_000, 500_000, 500_000, 300_000, 800_000],
        "type_local": ["Appartement", "Maison", "Maison", "Appartement", "Local commercial"],
        "surface_reelle_bati": [50, 120, 120, 70, 200],
        "nombre_pieces_principales": [2, 5, 5, 3, 0],
        "surface_terrain": [0, 500, 500, 0, 0],
        "code_departement": ["75", "31", "31", "13", "75"],
        "code_commune": ["75101", "31555", "31555", "13201", "75108"],
        "longitude": [2.35, 1.44, 1.44, 5.37, 2.34],
        "latitude": [48.85, 43.60, 43.60, 43.30, 48.87],
    })


def test_filter_nature_and_type(sample_df):
    df = cast_types(sample_df)
    out = filter_nature_and_type(df)
    assert (out["nature_mutation"] == "Vente").all()
    assert out["type_local"].isin(["Appartement", "Maison"]).all()
    assert len(out) == 3


def test_drop_missing(sample_df):
    """cast_types drops rows with NaN valeur_fonciere internally; drop_missing
    catches surface_reelle_bati / code_departement / date_mutation / type_local."""
    df = sample_df.copy()
    df.loc[0, "surface_reelle_bati"] = None
    df = cast_types(df)
    out = drop_missing(df)
    assert out["surface_reelle_bati"].notna().all()
    assert out["valeur_fonciere"].notna().all()


def test_filter_outliers_excludes_too_small_surface():
    df = pd.DataFrame({
        "valeur_fonciere": [200_000, 250_000],
        "surface_reelle_bati": [5.0, 60.0],
        "nombre_pieces_principales": [1, 3],
        "surface_terrain": [0.0, 0.0],
        "date_mutation": pd.to_datetime(["2024-01-01", "2024-02-01"]),
    })
    out = filter_outliers(df)
    assert len(out) == 1
    assert (out["surface_reelle_bati"] >= 9).all()


def test_filter_outliers_caps_pieces():
    """Test du nouveau cap pieces (problème #6 du diagnostic)."""
    df = pd.DataFrame({
        "valeur_fonciere": [200_000, 250_000, 300_000],
        "surface_reelle_bati": [50.0, 60.0, 70.0],
        "nombre_pieces_principales": [3, 5, 198],  # 198 = aberrant
        "surface_terrain": [0.0, 0.0, 0.0],
        "date_mutation": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
    })
    out = filter_outliers(df)
    assert len(out) == 2
    assert out["nombre_pieces_principales"].max() <= 15


def test_filter_outliers_caps_terrain_ratio():
    """Test du nouveau cap ratio terrain/bâti."""
    df = pd.DataFrame({
        "valeur_fonciere": [200_000, 250_000],
        "surface_reelle_bati": [50.0, 60.0],
        "nombre_pieces_principales": [3, 3],
        "surface_terrain": [100_000, 200],  # ratio 2000 = aberrant
        "date_mutation": pd.to_datetime(["2024-01-01", "2024-02-01"]),
    })
    out = filter_outliers(df)
    assert len(out) == 1


def test_filter_gps_france():
    df = pd.DataFrame({
        "code_departement": ["75", "75", "75"],
        "latitude": [48.85, 25.0, None],      # 1 valide, 1 hors France, 1 NaN
        "longitude": [2.35, 100.0, None],
    })
    out = filter_gps_france(df)
    # NaN gardé (sera imputé plus tard), Paris gardé, point hors France rejeté
    assert len(out) == 2


def test_dedupe_on_id_mutation(sample_df):
    df = cast_types(sample_df)
    df = filter_nature_and_type(df)
    out = dedupe(df)
    assert out["id_mutation"].is_unique


def test_prix_m2_computed_after_filter():
    df = pd.DataFrame({
        "valeur_fonciere": [500_000.0],
        "surface_reelle_bati": [100.0],
        "nombre_pieces_principales": [4],
        "surface_terrain": [0.0],
        "date_mutation": pd.to_datetime(["2024-01-01"]),
    })
    out = filter_outliers(df)
    assert "prix_m2" in out.columns
    assert out["prix_m2"].iloc[0] == 5000.0
