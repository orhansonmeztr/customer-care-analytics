"""Post-warranty service churn: will a car come back to the authorized network?

When the 3-year warranty ends, many owners move to cheaper independent garages. The
distributor and its dealers lose after-sales revenue and, more importantly, the contact
that leads to the next car sale.

Definition (agreed with the business before modelling): a vehicle *churns* if it has no
authorized service order in the 12 months after its warranty ends.

* The prediction is made ``LEAD_DAYS`` before the warranty ends, so there is time to act.
  Features use only events strictly before that cutoff, in training and in scoring alike;
  ``assert_no_leakage`` checks this.
* Validation is out-of-time: the model is trained on earlier warranty expiries and tested on
  later ones, the way it will be used.
* The contact threshold comes from costs, not from 0.5.
* Who *responds* to a retention contact is a causal question; this model ranks risk only.
  The targeting question is the subject of the uplift-decision-engine project.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .paths import Paths
from .quality import connect

NUMERIC = [
    "visits_before",
    "repairs_before",
    "days_since_last_visit",
    "avg_periodic_spend",
    "km_per_year",
    "nearest_service_km",
    "calls_180d",
    "complaints_180d",
    "csat_365d",
    "has_survey",
    "is_ev",
    "vehicle_age_at_cutoff",
]
CATEGORICAL = ["segment", "region"]
LEAD_DAYS = 90  # score cars this long before their warranty ends

# business assumptions for the retention call (change them in one place)
CONTACT_COST_TRY = 150.0  # agent time per call
VOUCHER_TRY = 1000.0  # maintenance voucher offered on the call
SAVE_RATE = 0.25  # share of would-be churners who come back because of the offer
REDEEM_RATE_STAYERS = 0.50  # customers who would have come anyway, and still use the voucher
ANNUAL_VALUE_TRY = 6000.0  # after-sales margin of a retained car, one year


def break_even_probability() -> float:
    """Churn probability above which a call pays for itself.

    Value of calling a car with churn probability p:
    p * SAVE_RATE * (VALUE - VOUCHER) - (1 - p) * REDEEM_RATE * VOUCHER - CONTACT_COST.
    The second term is why calling everyone is not free: loyal customers take the voucher too.
    """
    cost = CONTACT_COST_TRY + REDEEM_RATE_STAYERS * VOUCHER_TRY
    return cost / (SAVE_RATE * (ANNUAL_VALUE_TRY - VOUCHER_TRY) + REDEEM_RATE_STAYERS * VOUCHER_TRY)


def build_features(con, cohort: pd.DataFrame) -> pd.DataFrame:
    """``cohort`` has ``vin`` and ``cutoff``; every feature uses events before the cutoff."""
    con.register("cohort_df", cohort[["vin", "cutoff"]])
    return con.execute("""
    WITH c AS (
        SELECT k.vin, CAST(k.cutoff AS DATE) AS cutoff, v.customer_id, v.segment, v.is_ev,
               v.sale_date, v.warranty_end
        FROM cohort_df k JOIN dim_vehicle v USING (vin)
    ),
    so AS (
        SELECT c.vin,
            count(*) FILTER (WHERE s.order_type IN ('periodic', 'repair', 'warranty'))   AS visits_before,
            count(*) FILTER (WHERE s.order_type IN ('repair', 'warranty'))               AS repairs_before,
            date_diff('day', max(s.order_date), any_value(c.cutoff))                    AS days_since_last_visit,
            avg(s.amount_try) FILTER (WHERE s.order_type = 'periodic')                  AS avg_periodic_spend,
            max(s.odometer_km) / nullif(date_diff('day', any_value(c.sale_date),
                max(s.order_date) FILTER (WHERE s.odometer_km IS NOT NULL)), 0) * 365.25 AS km_per_year,
            max(s.order_date)                                                           AS last_event
        FROM c JOIN fact_service_orders s ON s.vin = c.vin AND s.order_date < c.cutoff
        WHERE NOT s.is_orphan
        GROUP BY c.vin
    ),
    calls AS (
        SELECT c.vin, count(*) AS calls_180d, max(f.arrived_at) AS last_event
        FROM c JOIN fact_calls f ON f.customer_id = c.customer_id
         AND f.call_date < c.cutoff AND f.call_date >= c.cutoff - INTERVAL 180 DAY
        GROUP BY c.vin
    ),
    tick AS (
        SELECT c.vin, count(*) AS complaints_180d, max(t.opened_at) AS last_event
        FROM c JOIN fact_tickets t ON t.customer_id = c.customer_id
         AND t.opened_date < c.cutoff AND t.opened_date >= c.cutoff - INTERVAL 180 DAY
        GROUP BY c.vin
    ),
    surv AS (
        SELECT c.vin, avg(s.csat) AS csat_365d, max(s.responded_at) AS last_event
        FROM c JOIN fact_calls f ON f.customer_id = c.customer_id
        JOIN fact_surveys s ON s.call_id = f.call_id
         AND s.responded_at < c.cutoff AND s.responded_at >= c.cutoff - INTERVAL 365 DAY
        GROUP BY c.vin
    )
    SELECT c.vin, c.cutoff, c.warranty_end, c.customer_id, c.segment, c.is_ev::INT AS is_ev, d.region,
        d.nearest_service_km,
        date_diff('day', c.sale_date, c.cutoff) / 365.25 AS vehicle_age_at_cutoff,
        coalesce(so.visits_before, 0) AS visits_before, coalesce(so.repairs_before, 0) AS repairs_before,
        so.days_since_last_visit, so.avg_periodic_spend, so.km_per_year,
        coalesce(calls.calls_180d, 0) AS calls_180d, coalesce(tick.complaints_180d, 0) AS complaints_180d,
        surv.csat_365d, (surv.csat_365d IS NOT NULL)::INT AS has_survey,
        greatest(so.last_event, CAST(calls.last_event AS DATE), CAST(tick.last_event AS DATE),
                 CAST(surv.last_event AS DATE)) AS latest_input_date
    FROM c
    LEFT JOIN so USING (vin) LEFT JOIN calls USING (vin) LEFT JOIN tick USING (vin)
    LEFT JOIN surv USING (vin)
    LEFT JOIN dim_customer d ON d.customer_id = c.customer_id
    """).df()


def assert_no_leakage(feats: pd.DataFrame) -> None:
    late = feats["latest_input_date"].notna() & (
        pd.to_datetime(feats["latest_input_date"]) >= pd.to_datetime(feats["cutoff"])
    )
    if late.any():
        raise AssertionError(f"{late.sum()} rows use events on or after their cutoff")


def labels(con, cohort: pd.DataFrame) -> pd.Series:
    con.register("lab_df", cohort[["vin"]])
    df = con.execute("""
        SELECT k.vin,
            NOT EXISTS (SELECT 1 FROM fact_service_orders s
                        WHERE s.vin = k.vin AND s.order_type <> 'adjustment'
                          AND s.order_date > v.warranty_end
                          AND s.order_date <= v.warranty_end + INTERVAL 365 DAY) AS churned
        FROM lab_df k JOIN dim_vehicle v USING (vin)
    """).df()
    return df.set_index("vin")["churned"].astype(int)


def training_cohort(con) -> pd.DataFrame:
    as_of, first_call = con.execute(
        "SELECT max(call_date), min(call_date) FROM fact_calls"
    ).fetchall()[0]
    hi = pd.Timestamp(as_of) - pd.Timedelta(days=365)  # the 12-month label window must be complete
    # the 180-day call/complaint windows before the cutoff must lie inside the call history
    lo = max(
        pd.Timestamp(first_call) + pd.Timedelta(days=180 + LEAD_DAYS), hi - pd.Timedelta(days=365)
    )
    df = con.execute(f"""
        SELECT vin, warranty_end, warranty_end - INTERVAL {LEAD_DAYS} DAY AS cutoff FROM dim_vehicle
        WHERE vin_valid AND NOT is_orphan
          AND warranty_end > DATE '{lo.date()}' AND warranty_end <= DATE '{hi.date()}'
        ORDER BY warranty_end
    """).df()
    return df


def lift_at(y: np.ndarray, p: np.ndarray, share: float) -> float:
    n = max(1, int(len(p) * share))
    top = np.argsort(-p)[:n]
    return float(y[top].mean() / y.mean())


def contact_policy(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Net value of contacting the top x% riskiest cars, on the test set."""
    order = np.argsort(-p)
    rows = []
    for share in np.arange(0.05, 1.0001, 0.05):
        n = int(round(len(p) * share))
        tp = y[order[:n]].sum()
        value = (
            tp * SAVE_RATE * (ANNUAL_VALUE_TRY - VOUCHER_TRY)
            - (n - tp) * REDEEM_RATE_STAYERS * VOUCHER_TRY
            - n * CONTACT_COST_TRY
        )
        rows.append(
            {
                "contact_share": round(share, 2),
                "contacted": n,
                "churners_reached": int(tp),
                "recall": tp / y.sum(),
                "net_value_try": value,
                "min_probability": float(p[order[n - 1]]) if n else 1.0,
            }
        )
    return pd.DataFrame(rows)


def _models():
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                NUMERIC,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    logit = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    gbm = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbose=-1,
        random_state=7,
        n_jobs=2,
    )
    return logit, gbm


def contributions(name: str, model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-car contribution of each original feature to the risk score (log-odds scale).

    LightGBM: SHAP values. Logistic regression: coefficient x standardized value, summed over
    the columns a feature expands into (one-hot levels, missing-value indicator).
    """
    feats = NUMERIC + CATEGORICAL
    if name == "lightgbm":
        return pd.DataFrame(
            model.predict(_as_lgb(X), pred_contrib=True)[:, :-1], columns=feats, index=X.index
        )
    pre, lr = model[0], model[-1]
    Xt = pre.transform(X[feats])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    raw = Xt * lr.coef_[0]
    out = pd.DataFrame(0.0, index=X.index, columns=feats)
    for j, col in enumerate(pre.get_feature_names_out()):
        part = col.split("__", 1)[1]
        part = part.removeprefix("missingindicator_")
        orig = part if part in NUMERIC else next(c for c in CATEGORICAL if part.startswith(c + "_"))
        out[orig] += raw[:, j]
    return out


def _as_lgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def run_churn(paths: Paths, test_share: float = 0.3) -> dict:
    con = connect(paths)
    cohort = training_cohort(con)
    feats = build_features(con, cohort)
    assert_no_leakage(feats)
    y_all = labels(con, cohort)
    feats["churned"] = feats["vin"].map(y_all)
    feats = feats.sort_values("cutoff").reset_index(drop=True)
    split = feats["cutoff"].iloc[int(len(feats) * (1 - test_share))]
    train, test = feats[feats["cutoff"] < split], feats[feats["cutoff"] >= split]

    logit, gbm = _models()
    logit.fit(train[NUMERIC + CATEGORICAL], train["churned"])
    gbm.fit(_as_lgb(train), train["churned"])
    y = test["churned"].to_numpy()
    preds = {
        "logistic_regression": logit.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1],
        "lightgbm": gbm.predict_proba(_as_lgb(test))[:, 1],
    }
    metrics = {}
    for name, p in preds.items():
        metrics[name] = {
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "lift_top10": lift_at(y, p, 0.10),
            "lift_top20": lift_at(y, p, 0.20),
        }
    best = max(metrics, key=lambda m: metrics[m]["pr_auc"])
    p_best = preds[best]
    policy = contact_policy(y, p_best)
    best_policy = policy.loc[policy["net_value_try"].idxmax()]

    fitted = {"logistic_regression": logit, "lightgbm": gbm}
    contrib = contributions(best, fitted[best], test)
    drivers = contrib.abs().mean()
    drivers = (drivers / drivers.sum()).sort_values(ascending=False)

    # score cars whose warranty ends in the next 6 months, features as of today
    as_of = pd.Timestamp(con.execute("SELECT max(call_date) FROM fact_calls").fetchall()[0][0])
    upcoming = con.execute(f"""
        SELECT vin FROM dim_vehicle WHERE vin_valid AND NOT is_orphan
          AND warranty_end > DATE '{(as_of + pd.Timedelta(days=LEAD_DAYS - 30)).date()}'
          AND warranty_end <= DATE '{(as_of + pd.Timedelta(days=LEAD_DAYS + 30)).date()}'
    """).df()
    upcoming["cutoff"] = as_of + pd.Timedelta(days=1)
    uf = build_features(con, upcoming)
    assert_no_leakage(uf)
    X_up = _as_lgb(uf) if best == "lightgbm" else uf[NUMERIC + CATEGORICAL]
    uf["churn_probability"] = fitted[best].predict_proba(X_up)[:, 1]
    uf["top_driver"] = contributions(best, fitted[best], uf).idxmax(axis=1).to_numpy()
    uf = uf.merge(con.execute("SELECT vin, model FROM dim_vehicle").df(), on="vin")
    con.close()
    uf["contact"] = uf["churn_probability"] >= best_policy["min_probability"]
    risk = uf[
        [
            "vin",
            "customer_id",
            "model",
            "segment",
            "region",
            "warranty_end",
            "churn_probability",
            "top_driver",
            "contact",
        ]
    ].sort_values("churn_probability", ascending=False)
    risk.to_parquet(paths.outputs / "churn_risk_upcoming.parquet", index=False)
    risk.to_csv(paths.exports / "churn_risk_upcoming.csv", index=False)
    test.assign(p=p_best).to_parquet(paths.outputs / "churn_test_scores.parquet", index=False)
    policy.to_parquet(paths.outputs / "churn_policy.parquet", index=False)
    drivers.rename("share").to_frame().to_parquet(paths.outputs / "churn_drivers.parquet")

    summary = {
        "cohort": len(feats),
        "train": len(train),
        "test": len(test),
        "train_period": [str(train["warranty_end"].min()), str(train["warranty_end"].max())],
        "test_period": [str(test["warranty_end"].min()), str(test["warranty_end"].max())],
        "churn_rate_train": float(train["churned"].mean()),
        "churn_rate_test": float(y.mean()),
        "metrics": {m: {k: round(v, 4) for k, v in d.items()} for m, d in metrics.items()},
        "best_model": best,
        "best_policy": {
            k: (float(v) if isinstance(v, np.floating | float) else int(v))
            for k, v in best_policy.items()
        },
        "break_even_probability": break_even_probability(),
        "call_everyone_value_try": float(policy["net_value_try"].iloc[-1]),
        "drivers_from": best,
        "top_drivers": drivers.head(6).round(3).to_dict(),
        "lead_days": LEAD_DAYS,
        "upcoming_scored": len(risk),
        "upcoming_to_contact": int(risk["contact"].sum()),
    }
    _write_report(paths, summary, policy)
    (paths.outputs / "churn.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _write_report(paths: Paths, s: dict, policy: pd.DataFrame) -> None:
    m = s["metrics"]
    bp = s["best_policy"]
    lines = [
        "# Post-warranty service churn",
        "",
        "**Churn:** no authorized service visit in the 12 months after the warranty ends.",
        "",
        f"**When:** each car is scored {s['lead_days']} days before its warranty ends, using only "
        "what was known on that day, so the retention team has time to call.",
        "",
        f"Cohort: {s['cohort']:,} cars whose warranty ended between {s['train_period'][0][:10]} and "
        f"{s['test_period'][1][:10]}. Out-of-time split: trained on {s['train']:,} earlier expiries, "
        f"tested on {s['test']:,} later ones. Churn rate {s['churn_rate_train']:.1%} (train), "
        f"{s['churn_rate_test']:.1%} (test).",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier | Lift top 10% | Lift top 20% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, d in m.items():
        lines.append(
            f"| {name.replace('_', ' ')} | {d['roc_auc']:.3f} | {d['pr_auc']:.3f} | "
            f"{d['brier']:.3f} | {d['lift_top10']:.2f}x | {d['lift_top20']:.2f}x |"
        )
    lines += [
        "",
        f"PR-AUC of a random ranking equals the churn rate ({s['churn_rate_test']:.3f}).",
        "",
        "## Whom to call",
        "",
        f"Assumptions: a call costs {CONTACT_COST_TRY:,.0f} TL and offers a {VOUCHER_TRY:,.0f} TL "
        f"maintenance voucher. It brings back {SAVE_RATE:.0%} of the churners it reaches, and a "
        f"retained car is worth {ANNUAL_VALUE_TRY:,.0f} TL of after-sales margin a year. "
        f"{REDEEM_RATE_STAYERS:.0%} of the customers who would have come back anyway also use the "
        "voucher: that is the cost of calling the wrong people. Break-even churn probability: "
        f"**{s['break_even_probability']:.1%}**.",
        "",
        f"Best policy on the test set: call the riskiest **{bp['contact_share']:.0%}** "
        f"({int(bp['contacted']):,} cars). It reaches {int(bp['churners_reached']):,} churners "
        f"({bp['recall']:.0%} of all) for an expected net value of **{bp['net_value_try']:,.0f} TL**. "
        "Calling everyone would be worth "
        f"{s['call_everyone_value_try']:,.0f} TL.",
        "",
        "| Call top | Cars | Churners reached | Net value (TL) |",
        "|---:|---:|---:|---:|",
    ]
    for r in policy.iloc[[0, 1, 3, 5, 7, 9, 13, 19]].itertuples():
        lines.append(
            f"| {r.contact_share:.0%} | {r.contacted:,} | {r.churners_reached:,} | "
            f"{r.net_value_try:,.0f} |"
        )
    lines += [
        "",
        f"## What drives the risk ({s['drivers_from'].replace('_', ' ')}, mean absolute "
        "contribution)",
        "",
        "| Feature | Share |",
        "|---|---:|",
    ]
    for f, v in s["top_drivers"].items():
        lines.append(f"| {f} | {v:.1%} |")
    lines += [
        "",
        f"**Today's call list:** {s['upcoming_scored']:,} cars reach the end of their warranty in "
        f"{s['lead_days'] - 30}-{s['lead_days'] + 30} days; {s['upcoming_to_contact']:,} are above "
        "the contact threshold "
        "(`exports/churn_risk_upcoming.csv`, with each car's main risk driver).",
        "",
        "The save rate is an assumption. The real number comes from a randomized test of the "
        "retention call, and the best targeting rule ranks cars by their *response* to the call "
        "(uplift), not by their risk.",
    ]
    (paths.reports / "churn.md").write_text("\n".join(lines) + "\n")
