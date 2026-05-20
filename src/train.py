"""Entraînement XGBoost avec validation croisée temporelle anti-fuite.

CHANGEMENT MAJEUR vs version initiale :
Les target encodings et tables d'enrichissement sont FITTÉS SUR LE TRAIN
de chaque fold puis appliqués au test. → R² réaliste.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.config import (
    CLEAN_PARQUET,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    TARGET,
    XGB_PARAMS,
)
from src.features import fit_features, transform_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values("date_mutation").reset_index(drop=True)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return X, y


def evaluate_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fold_num: int,
    n_splits: int,
) -> dict:
    """Évaluation propre d'un fold : fit features sur train, transform test."""
    logger.info(
        "--- Fold %d/%d : train=%d, test=%d ---",
        fold_num, n_splits, len(df_train), len(df_test),
    )

    # Fit features sur TRAIN uniquement
    train_feat, artifacts = fit_features(df_train.copy())
    # Transform TEST avec les artefacts du train
    test_feat = transform_features(df_test.copy(), artifacts)

    X_train, y_train = prepare_xy(train_feat)
    X_test, y_test = prepare_xy(test_feat)

    # XGBoost avec early stopping
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    true = np.expm1(y_test.values)

    rmse = float(np.sqrt(mean_squared_error(true, preds)))
    r2 = float(r2_score(true, preds))
    mape = float(np.mean(np.abs((true - preds) / true)) * 100)

    logger.info(
        "Fold %d : RMSE=%.1f €/m² | R²=%.3f | MAPE=%.2f%% (iter=%d)",
        fold_num, rmse, r2, mape, model.best_iteration or model.n_estimators,
    )
    return {
        "fold": fold_num,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "best_iter": int(model.best_iteration or model.n_estimators),
    }


def cross_validate_anti_leak(df: pd.DataFrame, n_splits: int = 5) -> list[dict]:
    """Validation croisée temporelle anti-fuite."""
    df = df.sort_values("date_mutation").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []

    for i, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        fold_metrics = evaluate_fold(df_train, df_test, i, n_splits)
        folds.append(fold_metrics)

    return folds


def train_final_model(df: pd.DataFrame) -> tuple[XGBRegressor, dict]:
    """Modèle final sur 100 % des données + artefacts pour inférence."""
    logger.info("=== ENTRAÎNEMENT FINAL sur %d lignes ===", len(df))

    # Pour le modèle final, on fit sur tout (pas de split)
    df_feat, artifacts = fit_features(df.copy())

    # Split temporel 90/10 pour le early stopping
    df_feat = df_feat.sort_values("date_mutation").reset_index(drop=True)
    cut = int(len(df_feat) * 0.9)
    df_train_feat = df_feat.iloc[:cut]
    df_val_feat = df_feat.iloc[cut:]

    X_train, y_train = prepare_xy(df_train_feat)
    X_val, y_val = prepare_xy(df_val_feat)

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info(
        "Modèle final entraîné (best_iter=%d sur %d)",
        model.best_iteration or model.n_estimators, XGB_PARAMS["n_estimators"],
    )
    return model, artifacts


def summarize_folds(folds: list[dict]) -> dict:
    rmse = [f["rmse"] for f in folds]
    r2 = [f["r2"] for f in folds]
    mape = [f["mape"] for f in folds]
    return {
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "mape_mean": float(np.mean(mape)),
        "mape_std": float(np.std(mape)),
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement XGBoost anti-fuite")
    parser.add_argument("--cv", type=int, default=5, help="Nombre de folds")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip CV, just train final model")
    args = parser.parse_args()

    logger.info("Lecture des données nettoyées : %s", CLEAN_PARQUET)
    df = pd.read_parquet(CLEAN_PARQUET)

    # Cross-validation (anti-fuite)
    metrics = {}
    if not args.skip_cv:
        folds = cross_validate_anti_leak(df, n_splits=args.cv)
        metrics = summarize_folds(folds)
        logger.info(
            "=== MOYENNES CV (anti-fuite) ===\n"
            "  RMSE = %.1f ± %.1f €/m²\n"
            "  R²   = %.3f ± %.3f\n"
            "  MAPE = %.2f ± %.2f %%",
            metrics["rmse_mean"], metrics["rmse_std"],
            metrics["r2_mean"], metrics["r2_std"],
            metrics["mape_mean"], metrics["mape_std"],
        )

    # Modèle final
    model, _ = train_final_model(df)
    joblib.dump(model, MODEL_PATH)
    logger.info("Modèle sauvegardé : %s", MODEL_PATH)

    if metrics:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Métriques : %s", METRICS_PATH)


if __name__ == "__main__":
    main()
