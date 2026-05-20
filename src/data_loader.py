"""Téléchargement multi-années des données DVF depuis data.gouv.fr."""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.config import DEFAULT_YEARS, DVF_BASE_URL, RAW_COLUMNS, RAW_DIR, RAW_PARQUET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_url(year: int) -> str:
    return f"{DVF_BASE_URL}/{year}/full.csv.gz"


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    logger.info("Téléchargement de %s", url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info("Sauvegardé : %s (%.1f Mo)", dest, dest.stat().st_size / 1e6)


def read_dvf_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or RAW_COLUMNS
    logger.info("Lecture : %s", path)
    open_fn = gzip.open if str(path).endswith(".gz") else open

    chunks = []
    with open_fn(path, "rt", encoding="utf-8") as f:
        for chunk in pd.read_csv(
            f,
            usecols=lambda c: c in columns,
            dtype={
                "code_postal": "string",
                "code_commune": "string",
                "code_departement": "string",
            },
            chunksize=200_000,
            low_memory=False,
        ):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    logger.info("Lignes : %d", len(df))
    return df


def load_year(year: int, force_download: bool = False) -> pd.DataFrame:
    raw_file = RAW_DIR / f"dvf_{year}.csv.gz"
    if not raw_file.exists() or force_download:
        url = build_url(year)
        download_file(url, raw_file)
    else:
        logger.info("Déjà présent : %s", raw_file)
    return read_dvf_csv(raw_file)


def load_multiple_years(years: list[int]) -> pd.DataFrame:
    dfs = []
    for year in years:
        try:
            df = load_year(year)
            df["annee_source"] = year
            dfs.append(df)
        except Exception as e:
            logger.warning("Année %d échouée : %s — on continue", year, e)
    if not dfs:
        raise RuntimeError("Aucune année n'a pu être téléchargée.")
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Combiné : %d lignes sur %d années (%s)",
        len(combined), len(dfs), [df["annee_source"].iloc[0] for df in dfs],
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Téléchargement DVF multi-années")
    parser.add_argument("--year", type=int, default=None,
                        help="Une seule année (override)")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Liste d'années (ex: --years 2022 2023 2024)")
    parser.add_argument("--force", action="store_true",
                        help="Re-télécharger même si présent")
    args = parser.parse_args()

    if args.year is not None:
        years = [args.year]
    elif args.years:
        years = args.years
    else:
        years = DEFAULT_YEARS
        logger.info("Aucune année spécifiée — utilise par défaut : %s", years)

    try:
        df = load_multiple_years(years)
        df.to_parquet(RAW_PARQUET, index=False)
        logger.info("Parquet brut écrit : %s (%d lignes)", RAW_PARQUET, len(df))
    except Exception as e:
        logger.error("Erreur : %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
