"""Customer care analytics dashboard.

Run the pipeline first (``care run-all``), then: ``streamlit run dashboard/app.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

DATA = Path(os.environ.get("CARE_DATA_DIR", "data"))
OUT = DATA / "outputs"

st.set_page_config(page_title="Customer Care Analytics", layout="wide")


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    with duckdb.connect(str(DATA / "warehouse.duckdb"), read_only=True) as con:
        return con.execute(sql).df()


@st.cache_data
def out(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")


def summary(name: str) -> dict:
    f = OUT / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


if not (DATA / "warehouse.duckdb").exists():
    st.error(f"No warehouse in `{DATA}`. Run `care run-all` first.")
    st.stop()

st.title("Customer Care Analytics & Planning")
st.caption(
    "Meridian Auto (fictional distributor) · synthetic data · all KPI definitions in "
    "docs/kpi_definitions.md"
)

daily = q("SELECT * FROM v_kpi_daily")
daily["call_date"] = pd.to_datetime(daily["call_date"])
min_d, max_d = daily["call_date"].min().date(), daily["call_date"].max().date()

with st.sidebar:
    st.header("Filters")
    period = st.date_input("Period", (max_d - pd.Timedelta(days=89), max_d), min_d, max_d)
    queues = st.multiselect(
        "Queue", sorted(daily["queue"].unique()), sorted(daily["queue"].unique())
    )
    st.caption("Days with an incomplete data load are excluded from KPIs.")

start, end = (pd.Timestamp(period[0]), pd.Timestamp(period[-1]))
sel = daily[daily["queue"].isin(queues) & ~daily["is_incomplete_load"].astype(bool)]


def kpis(df: pd.DataFrame) -> dict:
    eff = (df["offered"] - df["short_abandons"]).sum()
    surveys = df["surveys"].fillna(0)
    return {
        "Offered calls": df["offered"].sum(),
        "Service level (20 s)": df["answered_in_20s"].sum() / eff if eff else None,
        "Abandon rate": df["abandoned"].sum() / eff if eff else None,
        "AHT (min)": (df["aht_sec"] * df["answered"]).sum() / df["answered"].sum() / 60,
        "FCR (7 days)": (df["fcr_7d"] * df["answered"]).sum() / df["answered"].sum(),
        "CSAT top-2": (df["csat_top2"].fillna(0) * surveys).sum() / surveys.sum(),
        "NPS": (df["nps"].fillna(0) * surveys).sum() / surveys.sum(),
    }


tabs = st.tabs(
    [
        "Overview",
        "Forecast & staffing",
        "Segments",
        "Service churn",
        "Complaint alerts",
        "Data quality",
        "Ask the data",
    ]
)

with tabs[0]:
    cur = sel[(sel["call_date"] >= start) & (sel["call_date"] <= end)]
    span = end - start + pd.Timedelta(days=1)
    prev = sel[(sel["call_date"] >= start - span) & (sel["call_date"] < start)]
    k_cur, k_prev = kpis(cur), kpis(prev)
    cols = st.columns(len(k_cur))
    pct = {"Service level (20 s)", "Abandon rate", "FCR (7 days)", "CSAT top-2"}
    for col, (name, v) in zip(cols, k_cur.items(), strict=True):
        p = k_prev.get(name)
        if name in pct:
            col.metric(
                name,
                f"{v:.1%}",
                f"{(v - p) * 100:+.1f} pt" if p is not None else None,
                delta_color="inverse" if name == "Abandon rate" else "normal",
            )
        elif name == "Offered calls":
            col.metric(name, f"{v:,.0f}", f"{(v / p - 1):+.1%}" if p else None, delta_color="off")
        else:
            col.metric(
                name,
                f"{v:.1f}",
                f"{v - p:+.1f}" if p is not None else None,
                delta_color="inverse" if name.startswith("AHT") else "normal",
            )
    st.caption("Change vs the previous period of the same length. Service level target: 80%.")

    d = (
        cur.groupby("call_date")
        .agg(
            offered=("offered", "sum"),
            abandoned=("abandoned", "sum"),
            answered_in_20s=("answered_in_20s", "sum"),
            short=("short_abandons", "sum"),
        )
        .reset_index()
    )
    d["service_level"] = d["answered_in_20s"] / (d["offered"] - d["short"])
    c1, c2 = st.columns(2)
    c1.altair_chart(
        alt.Chart(d)
        .mark_line()
        .encode(x=alt.X("call_date:T", title=None), y=alt.Y("offered:Q", title="Offered calls"))
        .properties(title="Daily offered calls", height=280),
        width="stretch",
    )
    base = alt.Chart(d).encode(x=alt.X("call_date:T", title=None))
    c2.altair_chart(
        (
            base.mark_line().encode(
                y=alt.Y("service_level:Q", title="Service level", axis=alt.Axis(format="%"))
            )
            + alt.Chart(pd.DataFrame({"y": [0.8]}))
            .mark_rule(strokeDash=[4, 4], color="gray")
            .encode(y="y:Q")
        ).properties(title="Service level (target 80%)", height=280),
        width="stretch",
    )

    hm = q(f"""
        SELECT dayofweek(interval_start) AS dow, hour(interval_start) AS hour,
               count(*) FILTER (WHERE answered_in_20s) * 1.0
               / nullif(count(*) - count(*) FILTER (WHERE is_short_abandon), 0) AS service_level,
               count(*) AS calls
        FROM fact_calls
        WHERE call_date BETWEEN DATE '{start.date()}' AND DATE '{end.date()}'
          AND queue IN ({",".join(f"'{x}'" for x in queues) or "''"})
        GROUP BY ALL
    """)
    names = {1: "1 Mon", 2: "2 Tue", 3: "3 Wed", 4: "4 Thu", 5: "5 Fri", 6: "6 Sat", 0: "7 Sun"}
    hm["day"] = hm["dow"].map(names)
    st.altair_chart(
        alt.Chart(hm)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour"),
            y=alt.Y("day:O", title=None),
            color=alt.Color(
                "service_level:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0.5, 1]),
                legend=alt.Legend(format="%"),
            ),
            tooltip=["day", "hour", alt.Tooltip("service_level:Q", format=".1%"), "calls"],
        )
        .properties(title="Service level by weekday and hour", height=230),
        width="stretch",
    )

    reasons = q(f"""
        SELECT reason_code, count(*) AS calls, avg(repeat_7d::INT) AS repeat_rate
        FROM fact_calls WHERE is_answered
          AND call_date BETWEEN DATE '{start.date()}' AND DATE '{end.date()}'
          AND queue IN ({",".join(f"'{x}'" for x in queues) or "''"})
        GROUP BY 1 ORDER BY calls DESC LIMIT 12
    """)
    st.altair_chart(
        alt.Chart(reasons)
        .mark_bar()
        .encode(
            x=alt.X("calls:Q", title="Answered calls"),
            y=alt.Y("reason_code:N", sort="-x", title=None),
            color=alt.Color(
                "repeat_rate:Q",
                title="Repeat within 7 d",
                legend=alt.Legend(format="%"),
                scale=alt.Scale(scheme="oranges"),
            ),
            tooltip=["reason_code", "calls", alt.Tooltip("repeat_rate:Q", format=".1%")],
        )
        .properties(
            title="Call reasons (colour: share that calls again within 7 days)", height=320
        ),
        width="stretch",
    )

with tabs[1]:
    s = summary("forecast")
    if s:
        st.subheader("Backtest: monthly forecasts made at the start of each month")
        st.dataframe(
            pd.DataFrame(s["scores"]).style.format(
                {
                    "wape": "{:.1%}",
                    "bias": "{:+.1%}",
                    "wape_holiday_days": "{:.1%}",
                    "wape_normal_days": "{:.1%}",
                }
            ),
            hide_index=True,
        )
        st.caption(f"LightGBM P80 forecast covered {s['p80_coverage']:.0%} of actual days.")
        bt = out("forecast_backtest")
        long = bt.melt(
            id_vars="ds", value_vars=["y", "lgbm", "naive_7"], var_name="series", value_name="calls"
        )
        long["series"] = long["series"].map(
            {"y": "actual", "lgbm": "LightGBM", "naive_7": "seasonal naive"}
        )
        st.altair_chart(
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("ds:T", title=None),
                y=alt.Y("calls:Q", title="Calls per day"),
                color=alt.Color("series:N", title=None),
                strokeDash=alt.condition(
                    "datum.series == 'actual'", alt.value([1, 0]), alt.value([4, 3])
                ),
            )
            .properties(height=320),
            width="stretch",
        )

        st.subheader("Staffing plan (80/20, 30-minute intervals; Erlang C P50 / P80, Erlang A P80)")
        plan = out("staffing_plan")
        days = sorted(plan["ds"].dt.date.unique())
        day = st.selectbox("Day", days, index=0)
        p = plan[plan["ds"].dt.date == day]
        pl = p.melt(
            id_vars="interval_start",
            value_vars=[c for c in ("agents_p50", "agents_p80", "agents_p80_erlang_a") if c in p],
            var_name="plan",
            value_name="agents",
        )
        c1, c2 = st.columns([2, 1])
        c1.altair_chart(
            alt.Chart(pl)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X("interval_start:T", title=None, axis=alt.Axis(format="%H:%M")),
                y=alt.Y("agents:Q", title="Agents on the phone"),
                color=alt.Color("plan:N", title=None),
            )
            .properties(height=300),
            width="stretch",
        )
        c2.metric("Agent hours on the phone (P50)", f"{p['agents_p50'].sum() / 2:,.0f}")
        c2.metric(
            "Agents to schedule at peak (P80, 30% shrinkage)", f"{p['scheduled_p80'].max():,}"
        )
        curve = out("sl_curve")
        st.caption("Service level vs agents at the busiest interval of the plan")
        st.altair_chart(
            alt.Chart(curve)
            .mark_line(point=True)
            .encode(
                x=alt.X("agents:Q", title="Agents"),
                y=alt.Y("service_level:Q", title="Service level", axis=alt.Axis(format="%")),
                tooltip=[
                    "agents",
                    alt.Tooltip("service_level:Q", format=".1%"),
                    alt.Tooltip("asa_sec:Q", format=".0f"),
                    alt.Tooltip("occupancy:Q", format=".0%"),
                ],
            )
            .properties(height=240),
            width="stretch",
        )

with tabs[2]:
    seg = out("customer_segments")
    c1, c2 = st.columns(2)
    sizes = seg["rfm_segment"].value_counts().rename_axis("segment").reset_index(name="customers")
    c1.altair_chart(
        alt.Chart(sizes)
        .mark_bar()
        .encode(x="customers:Q", y=alt.Y("segment:N", sort="-x", title=None))
        .properties(title="RFM segments", height=260),
        width="stretch",
    )
    prof = (
        seg.groupby("cluster_name")
        .agg(
            customers=("customer_id", "count"),
            median_recency_days=("recency_days", "median"),
            visits_24m=("frequency_24m", "mean"),
            spend_24m=("monetary_24m", "median"),
            vehicle_age=("vehicle_age_years", "mean"),
            complaints_12m=("complaints_12m", "mean"),
        )
        .reset_index()
    )
    c2.dataframe(
        prof.style.format(
            {
                "median_recency_days": "{:.0f}",
                "visits_24m": "{:.1f}",
                "spend_24m": "{:,.0f}",
                "vehicle_age": "{:.1f}",
                "complaints_12m": "{:.2f}",
            }
        ),
        hide_index=True,
    )
    st.altair_chart(
        alt.Chart(seg.sample(min(4000, len(seg)), random_state=1))
        .mark_circle(size=14, opacity=0.5)
        .encode(
            x=alt.X("recency_days:Q", title="Days since last visit"),
            y=alt.Y("monetary_24m:Q", title="Spend, last 24 months (TL)"),
            color=alt.Color("rfm_segment:N", title="RFM segment"),
        )
        .properties(height=360),
        width="stretch",
    )

with tabs[3]:
    s = summary("churn")
    if s:
        m = s["metrics"][s["best_model"]]
        c = st.columns(4)
        c[0].metric("Test churn rate", f"{s['churn_rate_test']:.1%}")
        c[1].metric(f"ROC-AUC ({s['best_model']})", f"{m['roc_auc']:.3f}")
        c[2].metric("Lift in top 10%", f"{m['lift_top10']:.2f}x")
        c[3].metric("Cars on today's call list", f"{s['upcoming_to_contact']:,}")
        pol = out("churn_policy")
        st.altair_chart(
            alt.Chart(pol)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "contact_share:Q",
                    title="Share of cars called (riskiest first)",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("net_value_try:Q", title="Expected net value (TL)"),
                tooltip=[
                    "contacted",
                    "churners_reached",
                    alt.Tooltip("net_value_try:Q", format=",.0f"),
                ],
            )
            .properties(
                title="Value of the retention call by targeting depth (test set)", height=300
            ),
            width="stretch",
        )
        drv = out("churn_drivers").reset_index().rename(columns={"index": "feature"})
        st.altair_chart(
            alt.Chart(drv)
            .mark_bar()
            .encode(
                x=alt.X("share:Q", title="Share of mean |SHAP|", axis=alt.Axis(format="%")),
                y=alt.Y("feature:N", sort="-x", title=None),
            )
            .properties(title="Risk drivers", height=300),
            width="stretch",
        )
        st.subheader("Warranty ending in 60-120 days, highest risk first")
        st.dataframe(out("churn_risk_upcoming").head(200), hide_index=True)

with tabs[4]:
    weekly = out("complaint_weekly")
    alerts = out("complaint_alerts")
    st.dataframe(
        alerts[["week", "model", "fault_code", "n", "baseline", "q_value"]], hide_index=True
    )
    keys = sorted({(m, f) for m, f in zip(weekly["model"], weekly["fault_code"], strict=True)})
    default = keys.index(tuple(alerts.iloc[0][["model", "fault_code"]])) if len(alerts) else 0
    model, fault = st.selectbox(
        "Series", keys, index=default, format_func=lambda k: f"{k[0]} · {k[1]}"
    )
    w = weekly[(weekly["model"] == model) & (weekly["fault_code"] == fault)]
    base = alt.Chart(w).encode(x=alt.X("week:T", title=None))
    st.altair_chart(
        base.mark_bar(opacity=0.6).encode(
            y=alt.Y("n:Q", title="Tickets per week"),
            color=alt.Color(
                "alert:N", scale=alt.Scale(domain=[False, True], range=["#9ca3af", "#dc2626"])
            ),
        )
        + base.mark_line(color="black").encode(y="baseline:Q"),
        width="stretch",
    )
    cus_file = OUT / "complaint_cusum.parquet"
    if cus_file.exists():
        cus = out("complaint_cusum")
        cs = cus[(cus["model"] == model) & (cus["fault_code"] == fault)].dropna(subset=["cusum"])
        if len(cs):
            st.caption(
                "Daily CUSUM for the same series: it adds up small daily excesses over the "
                "baseline and alarms (red) when it crosses the threshold (dashed)."
            )
            cb = alt.Chart(cs).encode(x=alt.X("day:T", title=None))
            st.altair_chart(
                cb.mark_line().encode(y=alt.Y("cusum:Q", title="CUSUM"))
                + cb.mark_rule(strokeDash=[4, 4], color="gray").encode(y="mean(threshold):Q")
                + alt.Chart(cs[cs["alarm"]])
                .mark_point(color="#dc2626", size=60)
                .encode(x="day:T", y="cusum:Q"),
                width="stretch",
            )

with tabs[5]:
    s = summary("data_quality")
    if s:
        st.write(
            f"Extract date **{s['as_of']}** · timezone shift detected: "
            f"`{s['timezone_shift_hours']}`"
        )
        if s.get("entity_resolution"):
            er = s["entity_resolution"]
            st.write(
                f"Duplicate customers linked: {er['linked_pairs']:,} pairs · precision "
                f"{er['precision']:.1%} · recall {er['recall']:.1%} (vs injected truth)"
            )
        st.dataframe(
            pd.DataFrame(s["rules"])[
                [
                    "rule_id",
                    "table",
                    "dimension",
                    "severity",
                    "failed",
                    "total",
                    "failed_pct",
                    "action",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        load = q("SELECT * FROM dq_daily_load ORDER BY call_date")
        st.altair_chart(
            alt.Chart(load)
            .mark_line()
            .encode(x=alt.X("call_date:T", title=None), y=alt.Y("n_calls:Q", title="Calls loaded"))
            + alt.Chart(load[load["load_status"] != "ok"])
            .mark_point(color="red", size=80)
            .encode(
                x="call_date:T",
                y="n_calls:Q",
                tooltip=["call_date", "n_calls", "expected", "load_status"],
            ),
            width="stretch",
        )

with tabs[6]:
    from care_analytics.assistant.agent import Assistant
    from care_analytics.assistant.evaluate import load_golden
    from care_analytics.assistant.guard import connect_read_only
    from care_analytics.assistant.llm import default_model

    model = default_model()
    st.caption(
        f"Text-to-SQL over the star schema with `{model.name}`. Every query passes a guard: one "
        "read-only SELECT, allow-listed tables, no personal-data columns, a row limit, on a "
        "read-only connection. The default model runs on the local GPU node, so no data leaves "
        "the network; set ASSISTANT_PROVIDER=deepseek for the most accurate model."
    )
    examples = [g["question"] for g in load_golden()]
    pick = st.selectbox("Example questions", [""] + examples)
    question = st.text_input("Your question", value=pick)
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Writing and checking the query..."):
            con = connect_read_only(str(DATA / "warehouse.duckdb"))
            try:
                res = Assistant(con, model).ask(question, explain=True)
            except Exception as e:  # noqa: BLE001 - show model connection problems in the page
                st.error(f"The model could not be reached: {e}")
                st.stop()
            finally:
                con.close()
        icon = {"answered": "✅", "refused": "🛑", "blocked": "🛡️", "error": "⚠️"}[res.status]
        st.markdown(f"{icon} **{res.status}** · {res.attempts} attempt(s) · {res.seconds:.0f} s")
        if res.answer:
            st.info(res.answer)
        if res.sql:
            st.code(res.sql, language="sql")
        if res.rows is not None:
            st.dataframe(res.rows, hide_index=True)
        if res.error and res.status != "answered":
            st.caption(res.error)
