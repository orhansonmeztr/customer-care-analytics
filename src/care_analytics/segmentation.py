"""Customer segmentation for the authorized service business: RFM rules and k-means.

RFM is transparent and easy to act on. k-means finds groups RFM cannot see (vehicle age,
complaint behaviour). Both are profiled in business terms, because a segment nobody can
name or act on has no value.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .paths import Paths
from .quality import connect

KMEANS_FEATURES = [
    "recency_days",
    "frequency_24m",
    "log_monetary_24m",
    "vehicle_age_years",
    "calls_12m",
    "complaints_12m",
]

RFM_RULES = [  # (name, condition on r/f scores, suggested action)
    ("Loyal", lambda r, f: (r >= 4) & (f >= 4), "Loyalty benefits, first offer on a new car"),
    (
        "At risk (valuable)",
        lambda r, f: (r <= 2) & (f >= 4),
        "Personal call, maintenance offer before they move to an independent garage",
    ),
    ("New / growing", lambda r, f: (r >= 4) & (f <= 2), "Welcome journey, maintenance reminders"),
    (
        "Lapsed",
        lambda r, f: (r <= 2) & (f <= 2),
        "Low-cost reactivation campaign; test before scaling",
    ),
]


def customer_features(con) -> pd.DataFrame:
    as_of = con.execute("SELECT max(call_date) FROM fact_calls").fetchall()[0][0]
    df = con.execute(f"""
    WITH so AS (
        SELECT v.customer_id, s.*
        FROM fact_service_orders s JOIN dim_vehicle v USING (vin)
        WHERE NOT s.is_orphan AND NOT v.is_orphan AND s.amount_try IS NOT NULL
    ),
    rfm AS (
        SELECT customer_id,
            date_diff('day', max(order_date) FILTER (WHERE order_type <> 'adjustment'),
                      DATE '{as_of}')                                                AS recency_days,
            count(*) FILTER (WHERE order_type <> 'adjustment'
                             AND order_date > DATE '{as_of}' - INTERVAL 24 MONTH)       AS frequency_24m,
            coalesce(sum(amount_try) FILTER (WHERE order_date > DATE '{as_of}' - INTERVAL 24 MONTH), 0)
                                                                                    AS monetary_24m
        FROM so GROUP BY 1
        HAVING max(order_date) > DATE '{as_of}' - INTERVAL 36 MONTH
    ),
    veh AS (
        SELECT customer_id, count(*) AS n_vehicles,
            avg(date_diff('day', sale_date, DATE '{as_of}')) / 365.25 AS vehicle_age_years,
            bool_or(is_ev) AS has_ev, bool_or(segment = 'premium') AS has_premium
        FROM dim_vehicle WHERE NOT is_orphan GROUP BY 1
    ),
    calls AS (
        SELECT customer_id, count(*) AS calls_12m FROM fact_calls
        WHERE customer_id IS NOT NULL AND call_date > DATE '{as_of}' - INTERVAL 12 MONTH GROUP BY 1
    ),
    tick AS (
        SELECT customer_id, count(*) AS complaints_12m FROM fact_tickets
        WHERE customer_id IS NOT NULL AND opened_date > DATE '{as_of}' - INTERVAL 12 MONTH GROUP BY 1
    )
    SELECT r.*, v.n_vehicles, v.vehicle_age_years, v.has_ev, v.has_premium,
        coalesce(c.calls_12m, 0) AS calls_12m, coalesce(t.complaints_12m, 0) AS complaints_12m,
        d.region
    FROM rfm r JOIN veh v USING (customer_id)
    LEFT JOIN calls c USING (customer_id) LEFT JOIN tick t USING (customer_id)
    LEFT JOIN dim_customer d USING (customer_id)
    """).df()
    df["log_monetary_24m"] = np.log1p(df["monetary_24m"].clip(lower=0))
    return df


def rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # rank first so ties do not collapse quintiles; recent visits get the high score
    d["r"] = pd.qcut((-d["recency_days"]).rank(method="first"), 5, labels=False) + 1
    d["f"] = pd.qcut(d["frequency_24m"].rank(method="first"), 5, labels=False) + 1
    d["m"] = pd.qcut(d["monetary_24m"].rank(method="first"), 5, labels=False) + 1
    d["rfm_segment"] = "Middle"
    for name, cond, _ in RFM_RULES:
        d.loc[cond(d["r"], d["f"]) & (d["rfm_segment"] == "Middle"), "rfm_segment"] = name
    return d


def kmeans_segments(df: pd.DataFrame, k_range=range(3, 7), seed: int = 42):
    X = StandardScaler().fit_transform(df[KMEANS_FEATURES])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        scores[k] = float(silhouette_score(X[sample], km.labels_[sample]))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=KMEANS_FEATURES)
    names = {}
    pretty = {
        "recency_days": "recency",
        "frequency_24m": "visits",
        "log_monetary_24m": "spend",
        "vehicle_age_years": "vehicle age",
        "calls_12m": "calls",
        "complaints_12m": "complaints",
    }
    for c, row in centers.iterrows():
        top = row.abs().sort_values(ascending=False).index[:2]
        names[c] = " · ".join(f"{'high' if row[f] > 0 else 'low'} {pretty[f]}" for f in top)
    return labels, names, scores, best_k


def run_segmentation(paths: Paths) -> dict:
    con = connect(paths)
    df = customer_features(con)
    con.close()
    df = rfm_segments(df)
    labels, names, scores, k = kmeans_segments(df)
    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].map(names)
    df.to_parquet(paths.outputs / "customer_segments.parquet", index=False)
    df.drop(columns=["region"]).to_csv(paths.exports / "customer_segments.csv", index=False)

    agg = {
        "customers": ("customer_id", "count"),
        "recency_days": ("recency_days", "median"),
        "visits_24m": ("frequency_24m", "mean"),
        "spend_24m_try": ("monetary_24m", "median"),
        "vehicle_age": ("vehicle_age_years", "mean"),
        "calls_12m": ("calls_12m", "mean"),
        "complaints_12m": ("complaints_12m", "mean"),
    }
    rfm_prof = df.groupby("rfm_segment").agg(**agg).sort_values("customers", ascending=False)
    km_prof = df.groupby(["cluster", "cluster_name"]).agg(**agg).reset_index()
    actions = {n: a for n, _, a in RFM_RULES} | {"Middle": "Standard communication"}
    summary = {
        "customers": len(df),
        "k": k,
        "silhouette": {str(a): round(b, 3) for a, b in scores.items()},
        "rfm_sizes": rfm_prof["customers"].to_dict(),
    }
    lines = [
        "# Customer segmentation",
        "",
        f"Customers with an authorized service visit in the last 36 months: **{len(df):,}**.",
        "",
        "## RFM segments",
        "",
        "| Segment | Customers | Median days since visit | Visits / 24 m | Median spend 24 m (TL) "
        "| Suggested action |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rfm_prof.reset_index().itertuples():
        name = r.rfm_segment
        lines.append(
            f"| {name} | {r.customers:,} | {r.recency_days:.0f} | {r.visits_24m:.1f} | "
            f"{r.spend_24m_try:,.0f} | {actions[name]} |"
        )
    lines += [
        "",
        f"## k-means (k = {k}, chosen by silhouette)",
        "",
        "Silhouette by k: " + ", ".join(f"{a}: {b:.3f}" for a, b in scores.items()) + ".",
        "",
        "| Cluster | Profile | Customers | Median days since visit | Visits / 24 m | "
        "Vehicle age | Calls / 12 m | Complaints / 12 m |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in km_prof.itertuples():
        lines.append(
            f"| {r.cluster} | {r.cluster_name} | {r.customers:,} | {r.recency_days:.0f} | "
            f"{r.visits_24m:.1f} | {r.vehicle_age:.1f} | {r.calls_12m:.2f} | "
            f"{r.complaints_12m:.2f} |"
        )
    lines += [
        "",
        "Profiles are named from the two features that set each cluster apart (z-scores of "
        "the centroid). A segment is only useful if each one gets a different action; whether "
        "an action works is then measured with a controlled test.",
    ]
    (paths.reports / "segmentation.md").write_text("\n".join(lines) + "\n")
    (paths.outputs / "segmentation.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
