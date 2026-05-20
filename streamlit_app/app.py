"""Dashboard Streamlit pour estimation et exploration des prix immobiliers DVF."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLEAN_PARQUET, METRICS_PATH  # noqa: E402
from src.predict import BienImmobilier, PricePredictor  # noqa: E402

st.set_page_config(
    page_title="DVF Forecast",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_predictor() -> PricePredictor:
    return PricePredictor()


@st.cache_data(show_spinner="Chargement des données…")
def load_data() -> pd.DataFrame:
    import pyarrow.parquet as pq
    wanted = [
        "date_mutation", "type_local", "code_departement",
        "code_commune", "nom_commune", "prix_m2",
        "surface_reelle_bati", "valeur_fonciere",
        "latitude", "longitude",
    ]
    schema_cols = set(pq.read_schema(CLEAN_PARQUET).names)
    use = [c for c in wanted if c in schema_cols]
    df = pd.read_parquet(CLEAN_PARQUET, columns=use)
    if "date_mutation" in df.columns:
        df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    return df


@st.cache_data
def load_metrics() -> dict:
    import json
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# Sidebar
st.sidebar.title("🏠 DVF Forecast")
st.sidebar.markdown(
    "Estimation des prix immobiliers à partir des **Demandes de "
    "Valeurs Foncières** (DGFiP).\n\n"
    "Modèle XGBoost avec target encoding anti-fuite + clustering "
    "géographique K-means (~500 micro-zones)."
)
page = st.sidebar.radio(
    "Navigation",
    ["🎯 Estimateur", "📊 Exploration", "📈 Performance", "ℹ️ À propos"],
)

# =============================================================================
# PAGE : ESTIMATEUR
# =============================================================================
if page == "🎯 Estimateur":
    st.title("🎯 Estimateur de prix immobilier")
    st.markdown(
        "Renseignez les caractéristiques du bien. Les coordonnées GPS "
        "améliorent fortement la précision (micro-zone)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bien")
        type_local = st.selectbox("Type de bien", ["Appartement", "Maison"])
        surface = st.number_input("Surface habitable (m²)", 10, 500, 65)
        pieces = st.number_input("Nombre de pièces principales", 1, 15, 3)
        terrain = 0
        if type_local == "Maison":
            terrain = st.number_input("Surface terrain (m²)", 0, 5_000, 0, step=10)

    with col2:
        st.subheader("Localisation")
        try:
            predictor = load_predictor()
        except Exception as e:
            st.error(f"Modèle non disponible : {e}.\n\nLancez `run.bat all` d'abord.")
            st.stop()

        code_dept = st.text_input(
            "Code département (ex: 75, 13, 06)",
            value="75",
            max_chars=3,
        )
        code_commune = st.text_input(
            "Code commune INSEE (recommandé, 5 chiffres)",
            placeholder="ex: 75112",
        )

        st.markdown("**Coordonnées GPS** (très recommandé)")
        gps_col1, gps_col2 = st.columns(2)
        with gps_col1:
            latitude = st.number_input(
                "Latitude", min_value=41.0, max_value=51.5,
                value=48.85, step=0.001, format="%.4f",
            )
        with gps_col2:
            longitude = st.number_input(
                "Longitude", min_value=-5.5, max_value=10.0,
                value=2.35, step=0.001, format="%.4f",
            )

    if st.button("Estimer le prix", type="primary", use_container_width=True):
        bien = BienImmobilier(
            type_local=type_local,
            surface_reelle_bati=surface,
            nombre_pieces_principales=pieces,
            code_departement=code_dept.strip().zfill(2) if code_dept.isdigit() else code_dept,
            code_commune=code_commune.strip() or None,
            surface_terrain=terrain,
            latitude=latitude,
            longitude=longitude,
        )
        try:
            result = predictor.predict(bien)
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")
            st.stop()

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Prix total estimé",
            f"{result['prix_total_estime']:,.0f} €".replace(",", "\u202f"),
        )
        c2.metric(
            "Prix au m² estimé",
            f"{result['prix_m2_estime']:,.0f} €/m²".replace(",", "\u202f"),
        )
        c3.metric(
            "Micro-zone (K-means)",
            f"#{result['geo_cluster']}",
        )

        st.caption(
            "⚠️ Précision indicative ±15–25 %. Le DVF ne contient pas l'état "
            "du bien, l'étage, l'exposition ni la vue — facteurs qui peuvent "
            "représenter ±10–20 % supplémentaires."
        )

# =============================================================================
# PAGE : EXPLORATION
# =============================================================================
elif page == "📊 Exploration":
    st.title("📊 Exploration des données DVF")
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Impossible de charger les données : {e}")
        st.stop()

    df["annee"] = df["date_mutation"].dt.year if "date_mutation" in df.columns else 2024

    st.markdown(
        f"**{len(df):,}** transactions entre "
        f"**{df['date_mutation'].min().date()}** et "
        f"**{df['date_mutation'].max().date()}**.".replace(",", "\u202f")
    )

    col1, col2 = st.columns(2)
    with col1:
        types_sel = st.multiselect(
            "Type(s) de bien", df["type_local"].unique(),
            default=list(df["type_local"].unique()),
        )
    with col2:
        annees_sel = st.multiselect(
            "Année(s)", sorted(df["annee"].unique()),
            default=sorted(df["annee"].unique()),
        )

    df_f = df[df["type_local"].isin(types_sel) & df["annee"].isin(annees_sel)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df_f):,}".replace(",", "\u202f"))
    c2.metric("Prix médian/m²",
              f"{df_f['prix_m2'].median():,.0f} €".replace(",", "\u202f"))
    c3.metric("Surface médiane", f"{df_f['surface_reelle_bati'].median():.0f} m²")

    st.subheader("Distribution du prix au m²")
    sample = df_f.sample(min(50_000, len(df_f)), random_state=42)
    fig = px.histogram(
        sample, x="prix_m2", color="type_local", nbins=80,
        labels={"prix_m2": "Prix au m² (€)"}, opacity=0.7,
    )
    fig.update_layout(barmode="overlay", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prix médian par département")
    by_dept = (
        df_f.groupby("code_departement")["prix_m2"]
        .median().reset_index()
        .sort_values("prix_m2", ascending=False)
    )
    fig2 = px.bar(
        by_dept.head(30),
        x="code_departement", y="prix_m2",
        color="prix_m2", color_continuous_scale="RdYlGn_r",
        labels={"code_departement": "Département", "prix_m2": "Prix médian au m² (€)"},
    )
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Évolution annuelle du prix médian")
    df_time = df_f.copy()
    df_time["periode"] = df_time["date_mutation"].dt.to_period("Q").dt.to_timestamp()
    monthly = (
        df_time.groupby(["periode", "type_local"])["prix_m2"]
        .median().reset_index()
    )
    fig3 = px.line(
        monthly, x="periode", y="prix_m2", color="type_local",
        labels={"periode": "Trimestre", "prix_m2": "Prix médian au m² (€)"},
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
# PAGE : PERFORMANCE
# =============================================================================
elif page == "📈 Performance":
    st.title("📈 Performance du modèle (validation croisée anti-fuite)")
    metrics = load_metrics()

    if not metrics:
        st.warning("Aucune métrique disponible. Lancez `run.bat train`.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE moyen", f"{metrics['rmse_mean']:.0f} €/m²",
              f"±{metrics['rmse_std']:.0f}")
    c2.metric("R² moyen", f"{metrics['r2_mean']:.3f}",
              f"±{metrics['r2_std']:.3f}")
    c3.metric("MAPE moyen", f"{metrics['mape_mean']:.2f} %",
              f"±{metrics['mape_std']:.2f}")

    st.subheader("Détail par fold (TimeSeriesSplit)")
    folds_df = pd.DataFrame(metrics["folds"])
    st.dataframe(folds_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(folds_df, x="fold", y="rmse", markers=True,
                      title="RMSE par fold",
                      labels={"rmse": "RMSE (€/m²)", "fold": "Fold"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.line(folds_df, x="fold", y="r2", markers=True,
                       title="R² par fold",
                       labels={"r2": "R²", "fold": "Fold"})
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "ℹ️ Cette validation croisée est **anti-fuite** : les target "
        "encodings et tables d'enrichissement sont calculés uniquement "
        "sur le train de chaque fold, puis appliqués au test. Le R² "
        "affiché est donc une estimation honnête de la performance "
        "sur de nouvelles données."
    )

# =============================================================================
# PAGE : À PROPOS
# =============================================================================
else:
    st.title("ℹ️ À propos du projet")
    st.markdown("""
### DVF Forecast — Version corrigée

Pipeline de prévision des prix immobiliers sur les DVF (Demandes de Valeurs Foncières),
publiées par la DGFiP en open data.

#### Améliorations vs version initiale

1. **Multi-année** : 2020-2024 (vs 2024 seul) pour capturer les cycles
2. **Anti-fuite** : target encoding fitté sur train uniquement
3. **Micro-zones K-means** : ~500 clusters géographiques sur lat/lon (couverture 99 %)
4. **Aberrants** : filtre prix/m² strict [500 ; 20 000] + caps pieces/ratio terrain
5. **Winsorisation adaptative** : 2.5%/97.5% pour les dépts à forte hétérogénéité

#### Stack
Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly

#### Sources
- DVF : DGFiP — [data.gouv.fr](https://www.data.gouv.fr/) — Licence Ouverte 2.0

#### Auteur
Rayen Aissa — [github.com/rayenaissa/dvf-forecast](https://github.com/rayenaissa/dvf-forecast)
""")
