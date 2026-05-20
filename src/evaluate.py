"""Évaluation détaillée du modèle entraîné sur un hold-out temporel."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    CLEAN_PARQUET,
    COMMUNE_REVENUE_PATH,
    ENCODERS_PATH,
    FEATURE_COLUMNS,
    GEO_CLUSTERER_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    TARGET,
    TARGET_ENCODINGS_PATH,
)
from src.features import fit_features, transform_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_REPORT = PROCESSED_DIR / "evaluation_report.json"


def load_artifacts() -> dict:
    """Recharge les artefacts sauvegardés."""
    return {
        "encoders": joblib.load(ENCODERS_PATH),
        "target_encodings": joblib.load(TARGET_ENCODINGS_PATH),
        "geo_clusterer": joblib.load(GEO_CLUSTERER_PATH),
        "revenue_table": pd.read_parquet(COMMUNE_REVENUE_PATH),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def metrics_by_segment(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, by: str
) -> pd.DataFrame:
    df_eval = df.copy()
    df_eval["__true"] = y_true
    df_eval["__pred"] = y_pred
    rows = []
    for key, sub in df_eval.groupby(by):
        if len(sub) < 50:
            continue
        m = compute_metrics(sub["__true"].values, sub["__pred"].values)
        m[by] = key
        m["n"] = len(sub)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("rmse")


def main() -> None:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    logger.info("Lecture : %s", CLEAN_PARQUET)
    df = pd.read_parquet(CLEAN_PARQUET).sort_values("date_mutation").reset_index(drop=True)

    # Hold-out temporel = derniers 15%
    cut = int(len(df) * 0.85)
    df_train = df.iloc[:cut]
    df_test = df.iloc[cut:]
    logger.info("Hold-out : %d train / %d test", len(df_train), len(df_test))

    # Pour évaluer honnêtement : on re-fit les features sur le train,
    # et on applique au test (anti-fuite)
    _, artifacts = fit_features(df_train.copy())
    df_test_feat = transform_features(df_test.copy(), artifacts)

    model = joblib.load(MODEL_PATH)
    X = df_test_feat[FEATURE_COLUMNS]
    y_log = df_test_feat[TARGET].values

    preds_log = model.predict(X)
    y_true = np.expm1(y_log)
    y_pred = np.expm1(preds_log)

    overall = compute_metrics(y_true, y_pred)
    logger.info(
        "Hold-out : RMSE=%.1f €/m² | MAE=%.1f | R²=%.3f | MAPE=%.2f%%",
        overall["rmse"], overall["mae"], overall["r2"], overall["mape"],
    )

    by_type = metrics_by_segment(df_test_feat, y_true, y_pred, by="type_local")
    by_dept = metrics_by_segment(df_test_feat, y_true, y_pred, by="code_departement")

    logger.info("Par type :\n%s", by_type.to_string(index=False))
    logger.info("Top 10 dépts (RMSE) :\n%s",
                by_dept.head(10).to_string(index=False))

    report = {
        "overall": overall,
        "by_type": by_type.to_dict(orient="records"),
        "by_departement_best10": by_dept.head(10).to_dict(orient="records"),
        "by_departement_worst10": by_dept.tail(10).to_dict(orient="records"),
        "n_holdout": int(len(df_test)),
    }
    with open(EVAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Rapport : %s", EVAL_REPORT)


if __name__ == "__main__":
    main()
