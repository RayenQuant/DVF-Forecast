-- =============================================================================
-- Requêtes DuckDB pour exploration et agrégation des données DVF
-- Utilisation : duckdb -c ".read sql/queries.sql"
-- ou via Python : duckdb.sql(open('sql/queries.sql').read())
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Prix médian au m² par département
-- -----------------------------------------------------------------------------
SELECT
    code_departement,
    type_local,
    COUNT(*) AS nb_transactions,
    ROUND(MEDIAN(prix_m2), 0) AS prix_m2_median,
    ROUND(AVG(prix_m2), 0) AS prix_m2_moyen,
    ROUND(STDDEV(prix_m2), 0) AS prix_m2_std
FROM read_parquet('data/processed/dvf_features.parquet')
GROUP BY code_departement, type_local
ORDER BY prix_m2_median DESC;


-- -----------------------------------------------------------------------------
-- 2. Top 20 communes les plus chères (min. 50 ventes)
-- -----------------------------------------------------------------------------
SELECT
    code_commune,
    COUNT(*) AS nb_ventes,
    ROUND(MEDIAN(prix_m2), 0) AS prix_m2_median
FROM read_parquet('data/processed/dvf_features.parquet')
GROUP BY code_commune
HAVING nb_ventes >= 50
ORDER BY prix_m2_median DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- 3. Évolution mensuelle du prix médian au m²
-- -----------------------------------------------------------------------------
SELECT
    DATE_TRUNC('month', date_mutation) AS mois,
    type_local,
    COUNT(*) AS nb_transactions,
    ROUND(MEDIAN(prix_m2), 0) AS prix_m2_median
FROM read_parquet('data/processed/dvf_features.parquet')
GROUP BY mois, type_local
ORDER BY mois, type_local;


-- -----------------------------------------------------------------------------
-- 4. Distribution par tranche de surface
-- -----------------------------------------------------------------------------
SELECT
    CASE
        WHEN surface_reelle_bati < 30  THEN '<30 m²'
        WHEN surface_reelle_bati < 50  THEN '30-50 m²'
        WHEN surface_reelle_bati < 80  THEN '50-80 m²'
        WHEN surface_reelle_bati < 120 THEN '80-120 m²'
        ELSE '120+ m²'
    END AS tranche_surface,
    type_local,
    COUNT(*) AS nb,
    ROUND(MEDIAN(prix_m2), 0) AS prix_m2_median
FROM read_parquet('data/processed/dvf_features.parquet')
GROUP BY tranche_surface, type_local
ORDER BY type_local, tranche_surface;


-- -----------------------------------------------------------------------------
-- 5. Saisonnalité : prix médian par mois (toutes années confondues)
-- -----------------------------------------------------------------------------
SELECT
    mois,
    COUNT(*) AS nb_transactions,
    ROUND(MEDIAN(prix_m2), 0) AS prix_m2_median
FROM read_parquet('data/processed/dvf_features.parquet')
GROUP BY mois
ORDER BY mois;
