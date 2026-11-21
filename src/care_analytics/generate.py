"""Synthetic raw data for a fictional automotive distributor, "Meridian Auto".

The generator produces the kind of raw extracts a customer care analytics team receives:
CRM customers, vehicles, authorized service orders, call-center (PBX) records, complaint
tickets and post-call surveys. It deliberately injects the problems real extracts have,
so the data quality pipeline has something real to find:

* duplicate customers written by a second system (case, spacing, Turkish dotted/dotless I,
  phone formats)
* invalid VINs, vehicles pointing to missing customers
* duplicated service orders, odometer readings that go backwards or have extra zeros,
  negative amounts (some legitimate adjustments, some errors)
* a PBX migration: before 2023-06-01 the PBX wrote timestamps in UTC instead of local time
* one day missing from the load and one partially loaded day, replayed (duplicated) calls
* personal phone numbers typed into complaint descriptions

It also plants business events with known ground truth (a brake recall, a price increase,
a slowly growing EV battery complaint trend) so forecasting and anomaly detection can be
scored. Everything the generator injected is written to ``raw/_truth``; the pipeline never
reads it, only the evaluation code does.

All brands, models, names and numbers are fictional.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .calendar_tr import holiday_calendar
from .paths import Paths

FIRST_NAMES = [
    "Ahmet",
    "Mehmet",
    "Mustafa",
    "Ali",
    "Hüseyin",
    "Hasan",
    "İbrahim",
    "İsmail",
    "Osman",
    "Yusuf",
    "Murat",
    "Emre",
    "Burak",
    "Can",
    "Kerem",
    "Ayşe",
    "Fatma",
    "Emine",
    "Hatice",
    "Zeynep",
    "Elif",
    "Meryem",
    "Şerife",
    "Özlem",
    "Gül",
    "Derya",
    "Selin",
    "İrem",
    "Büşra",
    "Çağla",
]
LAST_NAMES = [
    "Yılmaz",
    "Kaya",
    "Demir",
    "Çelik",
    "Şahin",
    "Yıldız",
    "Yıldırım",
    "Öztürk",
    "Aydın",
    "Özdemir",
    "Arslan",
    "Doğan",
    "Kılıç",
    "Aslan",
    "Çetin",
    "Kara",
    "Koç",
    "Kurt",
    "Özkan",
    "Şimşek",
    "Polat",
    "Erdoğan",
    "Güneş",
    "Aksoy",
    "İnce",
    "Işık",
    "Uçar",
    "Tekin",
    "Akın",
    "Bulut",
]
# city, population weight, region
CITIES = [
    ("İstanbul", 0.30, "Marmara"),
    ("Ankara", 0.13, "İç Anadolu"),
    ("İzmir", 0.10, "Ege"),
    ("Bursa", 0.06, "Marmara"),
    ("Antalya", 0.06, "Akdeniz"),
    ("Kocaeli", 0.05, "Marmara"),
    ("Konya", 0.04, "İç Anadolu"),
    ("Adana", 0.04, "Akdeniz"),
    ("Kayseri", 0.03, "İç Anadolu"),
    ("Gaziantep", 0.03, "Güneydoğu"),
    ("Eskişehir", 0.03, "İç Anadolu"),
    ("Samsun", 0.03, "Karadeniz"),
    ("Trabzon", 0.03, "Karadeniz"),
    ("Diyarbakır", 0.03, "Güneydoğu"),
    ("Muğla", 0.04, "Ege"),
]
# model, brand, segment, is_ev, launch_year, sales weight, complaint multiplier, WMI prefix
MODELS = [
    ("Avra City", "Avra", "economy", False, 2014, 0.20, 1.0, "AVR"),
    ("Avra Cross", "Avra", "mid", False, 2017, 0.13, 0.9, "AVR"),
    ("Avra Cargo", "Avra", "commercial", False, 2014, 0.08, 1.0, "AVC"),
    ("Kestrel K3", "Kestrel", "mid", False, 2015, 0.16, 1.1, "KST"),
    ("Kestrel K5", "Kestrel", "mid", False, 2018, 0.11, 0.9, "KST"),
    ("Norda Tour", "Norda", "premium", False, 2016, 0.09, 0.8, "NRD"),
    ("Norda GT", "Norda", "premium", False, 2019, 0.06, 0.8, "NRD"),
    ("Volta E", "Volta", "mid", True, 2022, 0.11, 1.3, "VLT"),
    ("Volta E+", "Volta", "premium", True, 2023, 0.06, 1.0, "VLT"),
]
FAULT_CODES = {
    "ENG-01": "motor",
    "BRK-02": "fren",
    "ELC-03": "elektrik",
    "INF-04": "multimedya",
    "AC-05": "klima",
    "BAT-01": "batarya",
}
# queue: (share of calls, reasons, reason probabilities, mean talk seconds, transfer rate)
QUEUES = {
    "satis": (
        0.25,
        ["fiyat_bilgisi", "test_surusu", "kampanya", "teslimat_durumu"],
        [0.35, 0.20, 0.25, 0.20],
        230,
        0.06,
    ),
    "servis": (
        0.45,
        ["randevu", "bakim_fiyat", "yedek_parca", "servis_durumu", "geri_cagirma"],
        [0.40, 0.20, 0.15, 0.22, 0.03],
        280,
        0.07,
    ),
    "sikayet": (
        0.20,
        ["servis_kalitesi", "ariza_tekrari", "gecikme", "fatura_itirazi", "urun_kalitesi"],
        [0.25, 0.20, 0.20, 0.15, 0.20],
        400,
        0.15,
    ),
    "genel": (0.10, ["adres_iletisim", "kredi_finans", "diger"], [0.30, 0.40, 0.30], 170, 0.05),
}
SEGMENT_SPEND = {"economy": 4500, "mid": 6200, "premium": 11000, "commercial": 7000}
HOUR_PROFILE = np.array([0.5, 0.9, 1.3, 1.4, 1.2, 1.0, 1.1, 1.25, 1.15, 0.9, 0.6, 0.4])  # 08-19
OPEN_HOUR = 8
WEEKDAY_FACTOR = np.array([1.22, 1.10, 1.04, 1.00, 0.98, 0.70, 0.45])  # Mon..Sun
VIN_ALPHABET = np.array(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"))  # no I, O, Q

PBX_MIGRATION = pd.Timestamp("2023-06-01")
MISSING_DAY = pd.Timestamp("2024-09-17")
PARTIAL_DAY = pd.Timestamp("2024-08-14")
RECALL_START = pd.Timestamp("2024-03-11")
PRICE_INCREASE_START = pd.Timestamp("2024-12-23")
EV_ISSUE_START = pd.Timestamp("2025-02-01")


@dataclass(frozen=True)
class GenConfig:
    seed: int = 42
    scale: float = 1.0
    start: str = "2023-01-01"
    end: str = "2025-06-30"
    n_customers: int = 150_000
    daily_calls: float = 1_500.0
    n_service_points: int = 40

    @property
    def customers(self) -> int:
        return max(400, int(self.n_customers * self.scale))

    @property
    def calls_per_day(self) -> float:
        return max(30.0, self.daily_calls * self.scale)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _format_phone(digits: str, style: int) -> str:
    d = digits
    if style == 0:
        return f"0{d[:3]} {d[3:6]} {d[6:8]} {d[8:]}"
    if style == 1:
        return f"+90{d}"
    if style == 2:
        return d
    return f"0 ({d[:3]}) {d[3:6]} {d[6:8]} {d[8:]}"


def _random_vins(rng, prefixes: np.ndarray) -> np.ndarray:
    body = rng.choice(VIN_ALPHABET, size=(len(prefixes), 14))
    return np.array([p + "".join(b) for p, b in zip(prefixes, body, strict=True)])


def _times_in_day(rng, days: np.ndarray) -> np.ndarray:
    """Random arrival timestamps inside opening hours, following the intraday profile."""
    p = HOUR_PROFILE / HOUR_PROFILE.sum()
    hours = rng.choice(len(HOUR_PROFILE), size=len(days), p=p) + OPEN_HOUR
    secs = rng.integers(0, 3600, size=len(days))
    return days + pd.to_timedelta(hours * 3600 + secs, unit="s").to_numpy()


class _Generator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = pd.Timestamp(cfg.start)
        self.end = pd.Timestamp(cfg.end)
        self.end_ts = self.end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        self.truth: dict = {}

    # ------------------------------------------------------------------ master data
    def service_points(self) -> pd.DataFrame:
        rng = self.rng
        names, weights, regions = zip(*CITIES, strict=True)
        w = np.array(weights) / sum(weights)
        n = self.cfg.n_service_points
        # every city gets at least one service point, the rest follow population
        idx = np.r_[np.arange(len(names)), rng.choice(len(names), size=n - len(names), p=w)]
        sp = pd.DataFrame(
            {
                "service_code": [f"SRV-{i + 1:03d}" for i in range(n)],
                "city": np.array(names)[idx],
                "region": np.array(regions)[idx],
            }
        )
        sp["service_name"] = (
            sp["city"] + " Yetkili Servis " + (sp.groupby("city").cumcount() + 1).astype(str)
        )
        return sp[["service_code", "service_name", "city", "region"]]

    def customers(self, sp: pd.DataFrame) -> pd.DataFrame:
        rng, n = self.rng, self.cfg.customers
        names, weights, _ = zip(*CITIES, strict=True)
        w = np.array(weights) / sum(weights)
        city = np.array(names)[rng.choice(len(names), size=n, p=w)]
        big = np.isin(city, ["İstanbul", "Ankara", "İzmir"])
        distance = np.where(
            big, rng.lognormal(np.log(9), 0.5, n), rng.lognormal(np.log(16), 0.7, n)
        )
        sp_by_city = sp.groupby("city")["service_code"].apply(list).to_dict()
        service = np.array([sp_by_city[c][rng.integers(len(sp_by_city[c]))] for c in city])
        digits = np.array([f"5{x:09d}" for x in rng.integers(0, 10**9, n)])
        first = np.array(FIRST_NAMES)[rng.integers(len(FIRST_NAMES), size=n)]
        last = np.array(LAST_NAMES)[rng.integers(len(LAST_NAMES), size=n)]
        cust = pd.DataFrame(
            {
                "customer_id": np.arange(1, n + 1),
                "full_name": np.char.add(np.char.add(first, " "), last),
                "phone_digits": digits,
                "city": city,
                "nearest_service_km": distance.round(1),
                "home_service_code": service,
                "price_sensitivity": rng.beta(2, 2, n),  # hidden driver, never exported
            }
        )
        return cust

    def vehicles(self, cust: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        n_veh = np.where(rng.random(len(cust)) < 0.15, 2, 1)
        owner = np.repeat(cust["customer_id"].to_numpy(), n_veh)
        n = len(owner)
        # sales grow over time; sales start in 2016
        span = (self.end - pd.Timestamp("2016-01-01")).days - 15
        u = rng.random(n) ** 0.8
        sale = pd.Timestamp("2016-01-01") + pd.to_timedelta((u * span).astype(int), unit="D")
        m = pd.DataFrame(
            MODELS,
            columns=[
                "model",
                "brand",
                "segment",
                "is_ev",
                "launch_year",
                "weight",
                "complaint_mult",
                "wmi",
            ],
        )
        model_idx = rng.choice(len(m), size=n, p=m["weight"] / m["weight"].sum())
        # a model cannot be sold before it was launched: fall back to a conventional model
        too_early = sale.year.to_numpy() < m["launch_year"].to_numpy()[model_idx]
        conventional = m.index[~m["is_ev"]].to_numpy()
        model_idx[too_early] = rng.choice(conventional, size=too_early.sum())
        mm = m.iloc[model_idx].reset_index(drop=True)
        model_year = sale.year.to_numpy() + ((sale.month.to_numpy() >= 10) & (rng.random(n) < 0.4))
        veh = pd.DataFrame(
            {
                "vin": _random_vins(rng, mm["wmi"].to_numpy()),
                "customer_id": owner,
                "brand": mm["brand"],
                "model": mm["model"],
                "segment": mm["segment"],
                "is_ev": mm["is_ev"],
                "model_year": model_year,
                "sale_date": sale.normalize(),
                "km_per_year": np.clip(rng.normal(15000, 5000, n), 3000, 45000).round(),
                # hidden experience driver: how often this car and owner end up complaining
                "complaint_propensity": rng.gamma(2.0, 0.5, n) * mm["complaint_mult"].to_numpy(),
            }
        )
        return veh

    # ------------------------------------------------------------------ service history
    def service_orders(self, cust: pd.DataFrame, veh: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        c = cust.set_index("customer_id").loc[veh["customer_id"]]
        cp = veh["complaint_propensity"].to_numpy()
        cp_z = (cp - cp.mean()) / cp.std()
        seg = veh["segment"].to_numpy()
        logit = (
            1.05
            - 0.045 * c["nearest_service_km"].to_numpy()
            + np.select(
                [seg == "premium", seg == "mid", seg == "commercial"], [0.9, 0.35, 0.15], 0.0
            )
            - 2.2 * (c["price_sensitivity"].to_numpy() - 0.5)
            - 0.6 * cp_z
            + 0.35 * veh["is_ev"].to_numpy()
            + rng.normal(0, 0.55, len(veh))
        )
        p_post = _sigmoid(logit)
        p_in = 0.84 + 0.12 * p_post
        veh["p_post_warranty_visit"] = p_post  # hidden, used only for the truth file

        sale = veh["sale_date"].to_numpy()
        warranty_end = (veh["sale_date"] + pd.DateOffset(years=3)).to_numpy()
        rows = []
        for k in range(1, 11):
            # yearly periodic maintenance, about a month before each anniversary
            offset = 365 * k - 30 + rng.normal(0, 25, len(veh))
            date = sale + pd.to_timedelta(offset.round(), unit="D").to_numpy()
            in_warranty = date < warranty_end
            p = np.where(in_warranty, p_in, p_post)
            ok = (rng.random(len(veh)) < p) & (date <= np.datetime64(self.end))
            idx = np.flatnonzero(ok)
            rows.append(
                pd.DataFrame(
                    {
                        "veh_idx": idx,
                        "order_date": date[idx],
                        "order_type": "periodic",
                        "years": offset[idx] / 365,
                    }
                )
            )
        orders = pd.concat(rows, ignore_index=True)
        # some visits reveal a repair, a few weeks later
        rep = orders[rng.random(len(orders)) < 0.18].copy()
        rep["order_date"] = rep["order_date"] + pd.to_timedelta(
            rng.integers(7, 40, len(rep)), unit="D"
        )
        rep["years"] = rep["years"] + 0.06
        rep["order_type"] = np.where(
            rep["order_date"].to_numpy() < warranty_end[rep["veh_idx"]], "warranty", "repair"
        )
        orders = pd.concat([orders, rep[rep["order_date"] <= self.end]], ignore_index=True)

        v = veh.iloc[orders["veh_idx"]].reset_index(drop=True)
        base = v["segment"].map(SEGMENT_SPEND).to_numpy() * np.where(v["is_ev"], 0.7, 1.0)
        amount = np.where(
            orders["order_type"] == "periodic",
            rng.lognormal(np.log(base), 0.30),
            np.where(orders["order_type"] == "repair", rng.lognormal(np.log(base * 1.8), 0.6), 0.0),
        )
        # readings grow with time; the small noise per visit keeps them monotonic
        odo = (
            v["km_per_year"].to_numpy()
            * orders["years"].to_numpy()
            * rng.uniform(0.997, 1.003, len(v))
        )
        so = (
            pd.DataFrame(
                {
                    "vin": v["vin"],
                    "service_code": cust.set_index("customer_id")
                    .loc[v["customer_id"], "home_service_code"]
                    .to_numpy(),
                    "order_date": pd.to_datetime(orders["order_date"]).dt.normalize(),
                    "order_type": orders["order_type"],
                    "odometer_km": np.maximum(odo, 50).round().astype("int64"),
                    "amount_try": amount.round(2),
                }
            )
            .sort_values(["order_date", "vin"])
            .reset_index(drop=True)
        )

        # legitimate negative adjustments (refunds, goodwill credits)
        adj = so.sample(frac=0.005, random_state=int(rng.integers(1e9))).copy()
        adj["order_type"] = "adjustment"
        adj["amount_try"] = -(rng.uniform(200, 1500, len(adj))).round(2)
        so = pd.concat([so, adj], ignore_index=True)
        n = len(so)
        bad_neg = rng.random(n) < 0.0015
        so.loc[bad_neg & (so["order_type"] == "periodic"), "amount_try"] *= -1
        regress = rng.random(n) < 0.004
        so.loc[regress, "odometer_km"] = (so.loc[regress, "odometer_km"] * 0.55).astype("int64")
        typo = rng.random(n) < 0.002
        so.loc[typo, "odometer_km"] = so.loc[typo, "odometer_km"] * 100
        dup = so.sample(frac=0.01, random_state=int(rng.integers(1e9)))
        so = pd.concat([so, dup], ignore_index=True).sample(frac=1.0, random_state=1)
        so.insert(0, "order_id", np.arange(100001, 100001 + len(so)))
        self.truth["service_orders"] = {
            "exact_duplicates": int(len(dup)),
            "odometer_regressions": int(regress.sum()),
            "odometer_typos_x100": int(typo.sum()),
        }
        return so.sort_values("order_id").reset_index(drop=True)

    # ------------------------------------------------------------------ call center
    def calls(self, cust: pd.DataFrame, veh: pd.DataFrame, dup_ids: np.ndarray) -> pd.DataFrame:
        rng, cfg = self.rng, self.cfg
        cal = holiday_calendar(cfg.start, cfg.end)
        days = cal["date"].to_numpy()
        t = np.arange(len(cal))
        doy = cal["date"].dt.day_of_year.to_numpy()
        seasonal = (
            (1 + 0.08 * t / 365)
            * WEEKDAY_FACTOR[cal["date"].dt.day_of_week.to_numpy()]
            * (1 + 0.07 * np.sin(2 * np.pi * (doy - 80) / 365.25))
        )
        hol = np.select(
            [
                cal["holiday_kind"] == "religious",
                cal["holiday_kind"] == "national",
                cal["is_arife"],
                cal["is_post_holiday"],
            ],
            [0.30, 0.40, 0.70, 1.35],
            1.0,
        )
        expected = cfg.calls_per_day * seasonal * hol  # what the planners expect

        d_recall = (days - RECALL_START.to_datetime64()) / np.timedelta64(1, "D")
        d_price = (days - PRICE_INCREASE_START.to_datetime64()) / np.timedelta64(1, "D")
        d_ev = (days - EV_ISSUE_START.to_datetime64()) / np.timedelta64(1, "D")
        recall_mult = np.where((d_recall >= 0) & (d_recall < 35), 0.9 * np.exp(-d_recall / 8), 0.0)
        price_mult = np.where((d_price >= 0) & (d_price < 9), 0.6, 0.0)
        ev_ramp = np.where(d_ev >= 0, np.minimum(d_ev / 150, 1.0), 0.0)

        veh_w_complaint = veh["complaint_propensity"].to_numpy()
        is_k3 = (veh["model"] == "Kestrel K3").to_numpy()
        is_volta = (veh["model"] == "Volta E").to_numpy()
        noise = rng.gamma(150, 1 / 150, len(cal))
        streams = []  # (queue, lambda per day, fixed reason, vehicle weights)
        for q, (share, *_rest) in QUEUES.items():
            w = veh_w_complaint if q == "sikayet" else None
            streams.append((q, expected * share * noise, None, w))
        base_servis = expected * QUEUES["servis"][0]
        streams.append(("servis", base_servis * recall_mult, "geri_cagirma", is_k3.astype(float)))
        streams.append(("satis", expected * QUEUES["satis"][0] * price_mult, "fiyat_bilgisi", None))
        if is_volta.any():
            streams.append(
                (
                    "sikayet",
                    cfg.calls_per_day * 0.012 * ev_ramp,
                    "urun_kalitesi",
                    is_volta.astype(float),
                )
            )

        parts = []
        for q, lam, reason, w in streams:
            counts = rng.poisson(lam)
            day = np.repeat(days, counts)
            n = len(day)
            if n == 0:
                continue
            arrived = _times_in_day(rng, day)
            identified = rng.random(n) < (0.92 if reason else 0.70)
            if w is not None:
                vi = rng.choice(len(veh), size=n, p=w / w.sum())
            else:
                vi = rng.integers(len(veh), size=n)
            keep_vin = identified & (rng.random(n) < 0.85)
            cust_id = np.where(identified, veh["customer_id"].to_numpy()[vi], -1)
            vin = np.where(keep_vin, veh["vin"].to_numpy()[vi], None)
            # some callers are matched to the duplicate CRM record instead of the original one
            if len(dup_ids):
                use_dup = identified & (rng.random(n) < 0.03)
                cust_id = np.where(use_dup, rng.choice(dup_ids, size=n), cust_id)
                vin = np.where(use_dup, None, vin)
            if reason is None:
                reasons, probs = QUEUES[q][1], QUEUES[q][2]
                rc = np.array(reasons)[rng.choice(len(reasons), size=n, p=probs)]
            else:
                rc = np.full(n, reason)
            parts.append(
                pd.DataFrame(
                    {
                        "arrived_at": arrived,
                        "queue": q,
                        "reason_code": rc,
                        "customer_id": cust_id,
                        "vin": vin,
                    }
                )
            )
        calls = pd.concat(parts, ignore_index=True)

        # repeat calls: unresolved issues come back within a week
        p_rep = calls["queue"].map({"sikayet": 0.18, "servis": 0.10}).fillna(0.04).to_numpy()
        rep_mask = (calls["customer_id"].to_numpy() > 0) & (rng.random(len(calls)) < p_rep)
        rep = calls[rep_mask].copy()
        rep_day = rep["arrived_at"].dt.normalize() + pd.to_timedelta(
            rng.integers(1, 8, len(rep)), unit="D"
        )
        rep["arrived_at"] = _times_in_day(rng, rep_day.to_numpy())
        rep = rep[rep["arrived_at"] <= self.end_ts]
        calls["is_followup"] = False
        rep["is_followup"] = True
        calls = pd.concat([calls, rep], ignore_index=True)

        # queueing: staffing follows the planners' expectation, so unexpected load means waits
        day_idx = ((calls["arrived_at"].dt.normalize() - self.start).dt.days).to_numpy()
        actual = np.bincount(day_idx, minlength=len(cal))
        planned = expected * 1.10  # planners include the usual repeat calls
        ratio = actual[day_idx] / planned[day_idx]
        hour = calls["arrived_at"].dt.hour.to_numpy()
        hour_f = (HOUR_PROFILE[np.clip(hour - OPEN_HOUR, 0, 11)] / HOUR_PROFILE.mean()) ** 0.5
        mean_wait = np.clip(8 * np.exp(5 * (ratio - 0.95)) * hour_f, 3, 900)
        n = len(calls)
        wait = rng.exponential(mean_wait)
        patience = rng.exponential(110, n) + 4
        answered = wait <= patience
        talk_mean = calls["queue"].map({k: v[3] for k, v in QUEUES.items()}).to_numpy()
        talk = rng.lognormal(np.log(talk_mean) - 0.18, 0.6)
        hold = np.where(rng.random(n) < 0.3, rng.exponential(40, n), 0).round()
        wrap = np.clip(rng.normal(25, 10, n), 5, None).round()
        arrived = calls["arrived_at"]
        answered_at = arrived + pd.to_timedelta(np.where(answered, wait, np.nan), unit="s")
        ended_at = np.where(
            answered,
            answered_at + pd.to_timedelta(talk + hold, unit="s"),
            arrived + pd.to_timedelta(patience, unit="s"),
        )
        transfer_p = calls["queue"].map({k: v[4] for k, v in QUEUES.items()}).to_numpy()
        reason = calls["reason_code"].where(answered)
        reason = reason.where(rng.random(n) > 0.04)  # agents skip the reason code sometimes

        calls = (
            calls.assign(
                answered_at=answered_at.dt.floor("s"),
                ended_at=pd.to_datetime(ended_at).floor("s"),
                arrived_at=arrived.dt.floor("s"),
                hold_sec=np.where(answered, hold, 0).astype("int64"),
                wrap_sec=np.where(answered, wrap, 0).astype("int64"),
                reason_code=reason,
                agent_id=np.where(
                    answered,
                    np.char.add("AG", np.char.zfill(rng.integers(1, 151, n).astype(str), 3)),
                    None,
                ),
                transferred=answered & (rng.random(n) < transfer_p),
                wait_sec_true=wait,
            )
            .sort_values("arrived_at")
            .reset_index(drop=True)
        )
        calls.insert(0, "call_id", np.arange(5_000_001, 5_000_001 + len(calls)))
        calls["customer_id"] = calls["customer_id"].where(calls["customer_id"] > 0).astype("Int64")
        return calls

    def tickets(self, calls: pd.DataFrame, cust: pd.DataFrame, veh: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        vinfo = veh.set_index("vin")
        src = calls[
            (calls["queue"] == "sikayet") & calls["answered_at"].notna() & calls["vin"].notna()
        ]
        src = src[rng.random(len(src)) < 0.55]
        t_calls = pd.DataFrame(
            {
                "opened_at": src["ended_at"],
                "channel": "call",
                "customer_id": src["customer_id"],
                "vin": src["vin"],
                "category": src["reason_code"].fillna("servis_kalitesi"),
                "call_id": src["call_id"],
            }
        )
        # background tickets from the web form and dealers, weighted by complaint propensity
        days = pd.date_range(self.cfg.start, self.cfg.end, freq="D")
        w = veh["complaint_propensity"].to_numpy()
        lam_bg = max(2.0, self.cfg.calls_per_day * 0.03)
        k = rng.poisson(lam_bg, len(days))
        vi = rng.choice(len(veh), size=k.sum(), p=w / w.sum())
        cats = list(QUEUES["sikayet"][1])
        t_bg = pd.DataFrame(
            {
                "opened_at": _times_in_day(rng, np.repeat(days.to_numpy(), k)),
                "channel": rng.choice(["web", "dealer"], size=k.sum()),
                "customer_id": veh["customer_id"].to_numpy()[vi],
                "vin": veh["vin"].to_numpy()[vi],
                "category": rng.choice(cats, size=k.sum(), p=QUEUES["sikayet"][2]),
            }
        )
        tickets = pd.concat([t_calls, t_bg], ignore_index=True)

        is_ev = vinfo.loc[tickets["vin"], "is_ev"].to_numpy()
        codes = ["ENG-01", "BRK-02", "ELC-03", "INF-04", "AC-05"]
        fault = np.array(codes)[
            rng.choice(len(codes), size=len(tickets), p=[0.25, 0.15, 0.25, 0.2, 0.15])
        ]
        fault = np.where(is_ev & (rng.random(len(tickets)) < 0.3), "BAT-01", fault)
        fault = np.where(is_ev & (fault == "ENG-01"), "ELC-03", fault)
        # only product complaints carry a fault code
        product = tickets["category"].isin(["urun_kalitesi", "ariza_tekrari"]).to_numpy()
        tickets["fault_code"] = np.where(product, fault, None)
        planted = []

        # planted anomaly 1: brake complaints on Kestrel K3 around the recall
        k3 = veh.index[veh["model"] == "Kestrel K3"].to_numpy()
        d = (days - RECALL_START).days.to_numpy()
        lam = np.where((d >= -10) & (d < 40), 0.9 * lam_bg * np.exp(-np.abs(d - 5) / 12), 0)
        planted.append(self._planted(days, lam, k3, veh, "BRK-02", "Kestrel K3"))
        # planted anomaly 2: slowly rising EV battery complaints on Volta E
        ve = veh.index[veh["model"] == "Volta E"].to_numpy()
        d = (days - EV_ISSUE_START).days.to_numpy()
        lam = np.where(d >= 0, 0.5 * lam_bg * np.minimum(d / 150, 1.0), 0)
        planted.append(self._planted(days, lam, ve, veh, "BAT-01", "Volta E"))
        tickets = pd.concat([tickets] + [p for p in planted if len(p)], ignore_index=True)

        cust_service = cust.set_index("customer_id")["home_service_code"]
        cid = tickets["customer_id"].astype("Int64")
        tickets["service_code"] = cust_service.reindex(cid.fillna(-1).astype(int)).to_numpy()
        tickets["opened_at"] = pd.to_datetime(tickets["opened_at"]).dt.floor("s")
        dur = pd.to_timedelta(rng.lognormal(np.log(3.0 * 86400), 0.9, len(tickets)), unit="s")
        closed = tickets["opened_at"] + dur
        tickets["closed_at"] = closed.where(closed <= self.end_ts).dt.floor("s")
        tickets["status"] = np.where(tickets["closed_at"].isna(), "acik", "kapali")
        tickets["description"] = self._descriptions(
            tickets, vinfo.loc[tickets["vin"], "model"].to_numpy()
        )
        tickets = tickets.sample(frac=1.0, random_state=3).sort_values("opened_at")
        tickets.insert(0, "ticket_id", np.arange(900001, 900001 + len(tickets)))
        tickets["customer_id"] = tickets["customer_id"].astype("Int64")
        tickets["call_id"] = tickets["call_id"].astype("Int64")
        return tickets[
            [
                "ticket_id",
                "opened_at",
                "closed_at",
                "status",
                "channel",
                "customer_id",
                "vin",
                "service_code",
                "category",
                "fault_code",
                "call_id",
                "description",
            ]
        ]

    def _planted(self, days, lam, pool, veh, fault, model):
        rng = self.rng
        if len(pool) == 0:
            return pd.DataFrame()
        k = rng.poisson(lam)
        if k.sum() == 0:
            return pd.DataFrame()
        vi = rng.choice(pool, size=k.sum())
        self.truth.setdefault("planted_anomalies", []).append(
            {
                "model": model,
                "fault_code": fault,
                "onset": str(days[np.flatnonzero(lam > 0)[0]].date()),
                "extra_tickets": int(k.sum()),
            }
        )
        return pd.DataFrame(
            {
                "opened_at": _times_in_day(rng, np.repeat(days.to_numpy(), k)),
                "channel": rng.choice(["web", "dealer", "call"], size=k.sum(), p=[0.4, 0.4, 0.2]),
                "customer_id": veh["customer_id"].to_numpy()[vi],
                "vin": veh["vin"].to_numpy()[vi],
                "category": "urun_kalitesi",
                "fault_code": fault,
            }
        )

    def _descriptions(self, tickets: pd.DataFrame, models: np.ndarray) -> list[str]:
        rng = self.rng
        templates = {
            "servis_kalitesi": "{m} aracımın bakımı sonrası servis ilgisinden memnun kalmadım.",
            "ariza_tekrari": "{m} aracımda {f} arızası tekrar etti, ikinci kez servise gidiyorum.",
            "gecikme": "{m} aracım için verilen teslim tarihi geçti, bilgi alamıyorum.",
            "fatura_itirazi": "{m} bakım faturasında onaylamadığım kalemler var.",
            "urun_kalitesi": "{m} aracımda {f} ile ilgili bir sorun yaşıyorum.",
        }
        out = []
        for cat, f, m in zip(tickets["category"], tickets["fault_code"], models, strict=True):
            text = templates.get(cat, templates["servis_kalitesi"]).format(
                m=m, f=FAULT_CODES.get(f, "genel") if isinstance(f, str) else "genel"
            )
            if rng.random() < 0.02:  # customers type their phone number into free text
                text += (
                    f" Beni 05{rng.integers(30, 56)} {rng.integers(100, 999)} "
                    f"{rng.integers(10, 99)} {rng.integers(10, 99)} numaradan arayın."
                )
            out.append(text)
        return out

    def surveys(self, calls: pd.DataFrame, veh: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        c = calls[calls["answered_at"].notna() & calls["customer_id"].notna()].copy()
        cp = veh.set_index("vin")["complaint_propensity"]
        cp_call = cp.reindex(c["vin"]).fillna(cp.mean()).to_numpy()
        sat = (
            4.35
            - 0.004 * np.minimum(c["wait_sec_true"].to_numpy(), 600)
            - 0.6 * (c["queue"] == "sikayet").to_numpy()
            - 0.5 * c["is_followup"].to_numpy()
            - 0.35 * (cp_call - cp.mean()) / cp.std()
            + rng.normal(0, 0.75, len(c))
        )
        # unhappy customers answer surveys less often (missing not at random)
        responded = rng.random(len(c)) < np.where(sat > 3.5, 0.17, 0.09)
        c, sat = c[responded], sat[responded]
        s = pd.DataFrame(
            {
                "survey_id": np.arange(700001, 700001 + len(c)),
                "call_id": c["call_id"].to_numpy(),
                "responded_at": (
                    c["ended_at"] + pd.to_timedelta(rng.uniform(0.01, 2, len(c)), unit="D")
                )
                .dt.floor("s")
                .to_numpy(),
                "csat": np.clip(np.round(sat), 1, 5).astype(int),
                "nps": np.clip(
                    np.round(2.1 * sat - 0.9 + rng.normal(0, 1.0, len(c))), 0, 10
                ).astype(int),
            }
        )
        orphan = rng.random(len(s)) < 0.002
        s.loc[orphan, "call_id"] = s.loc[orphan, "call_id"] + 90_000_000
        return s

    # ------------------------------------------------------------------ raw extracts
    def raw_customers(
        self, cust: pd.DataFrame, veh: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        rng = self.rng
        n = len(cust)
        first_sale = veh.groupby("customer_id")["sale_date"].min()
        created = (
            first_sale.reindex(cust["customer_id"]).to_numpy()
            - pd.to_timedelta(rng.integers(0, 30, n), unit="D").to_numpy()
        )
        raw = pd.DataFrame(
            {
                "customer_id": cust["customer_id"],
                "full_name": cust["full_name"],
                "phone": [
                    _format_phone(d, s)
                    for d, s in zip(cust["phone_digits"], rng.integers(0, 4, n), strict=True)
                ],
                "email": [
                    self._email(nm, i)
                    for nm, i in zip(cust["full_name"], cust["customer_id"], strict=True)
                ],
                "city": cust["city"],
                "nearest_service_km": cust["nearest_service_km"],
                "created_at": pd.to_datetime(created),
                "source_system": "crm",
            }
        )
        raw.loc[rng.random(n) < 0.15, "email"] = None
        raw.loc[rng.random(n) < 0.01, "email"] = "eksik@"
        raw.loc[rng.random(n) < 0.02, "phone"] = None
        raw.loc[rng.random(n) < 0.008, "phone"] = "123"
        raw.loc[rng.random(n) < 0.002, "created_at"] = pd.Timestamp("2030-01-01")
        raw.loc[rng.random(n) < 0.003, "created_at"] = pd.Timestamp("1900-01-01")
        city_variant = rng.random(n) < 0.25
        raw.loc[city_variant, "city"] = [
            self._city_variant(c) for c in raw.loc[city_variant, "city"]
        ]

        # duplicates written by the dealer portal: same person, messier fields, new id
        dup_src = raw.sample(frac=0.03, random_state=int(rng.integers(1e9)))
        dup = dup_src.copy()
        dup["customer_id"] = np.arange(n + 1, n + 1 + len(dup))
        dup["source_system"] = "dealer_portal"
        dup["full_name"] = [self._name_variant(x) for x in dup["full_name"]]
        digits = cust.set_index("customer_id").loc[dup_src["customer_id"], "phone_digits"]
        dup["phone"] = [
            _format_phone(d, s) for d, s in zip(digits, rng.integers(0, 4, len(dup)), strict=True)
        ]
        dup["email"] = dup["email"].where(rng.random(len(dup)) < 0.6).str.upper()
        dup["city"] = [self._city_variant(c) for c in dup["city"]]
        dup["created_at"] = dup["created_at"] + pd.to_timedelta(
            rng.integers(30, 900, len(dup)), unit="D"
        )
        self.truth["duplicate_customers"] = pd.DataFrame(
            {"original_id": dup_src["customer_id"].to_numpy(), "duplicate_id": dup["customer_id"]}
        )
        raw = pd.concat([raw, dup], ignore_index=True)
        raw.loc[rng.random(len(raw)) < 0.001, "full_name"] = None
        return raw, dup["customer_id"].to_numpy()

    def _email(self, name: str, i: int) -> str:
        tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        user = name.translate(tr).lower().replace(" ", ".")
        return f"{user}{i}@{['mail.com', 'posta.net', 'eposta.com'][i % 3]}"

    def _city_variant(self, city: str) -> str:
        r = self.rng.integers(4)
        if r == 0:
            return city.upper()  # Python upper: "İzmir" -> "İZMIR", "istanbul" stays dotless issue
        if r == 1:
            return city.lower()  # "İstanbul".lower() -> "i̇stanbul" with a combining dot
        if r == 2:
            return city.translate(str.maketrans("İıŞşĞğÜüÖöÇç", "IiSsGgUuOoCc"))
        return f" {city} "

    def _name_variant(self, name: str) -> str:
        r = self.rng.integers(3)
        if r == 0:
            return name.upper() + " "  # "Ali Kaya".upper() -> "ALI KAYA", not "ALİ KAYA"
        if r == 1:
            return "  ".join(name.split())
        return name.translate(str.maketrans("İıŞşĞğÜüÖöÇç", "IiSsGgUuOoCc"))

    def raw_vehicles(self, veh: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        raw = veh[["vin", "customer_id", "brand", "model", "model_year", "sale_date"]].copy()
        raw["warranty_months"] = 36
        n = len(raw)
        short = rng.random(n) < 0.003
        raw.loc[short, "vin"] = raw.loc[short, "vin"].str[:16]
        bad_char = rng.random(n) < 0.002
        raw.loc[bad_char, "vin"] = (
            raw.loc[bad_char, "vin"].str[:9] + "O" + raw.loc[bad_char, "vin"].str[10:]
        )
        orphan = rng.random(n) < 0.004
        raw.loc[orphan, "customer_id"] = raw.loc[orphan, "customer_id"] + 10_000_000
        self.truth["vehicles"] = {
            "invalid_vin": int(short.sum() + bad_char.sum()),
            "orphan_customer": int(orphan.sum()),
        }
        return raw

    def raw_calls(self, calls: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        raw = calls.drop(columns=["wait_sec_true", "is_followup"]).copy()
        local_day = raw["arrived_at"].dt.normalize()
        raw["source_system"] = np.where(raw["arrived_at"] < PBX_MIGRATION, "pbx_v1", "pbx_v2")
        v1 = raw["source_system"] == "pbx_v1"
        for col in ("arrived_at", "answered_at", "ended_at"):
            raw.loc[v1, col] = raw.loc[v1, col] - pd.Timedelta(hours=3)  # v1 wrote UTC
        broken = rng.random(len(raw)) < 0.0003
        raw.loc[broken, "ended_at"] = raw.loc[broken, "arrived_at"] - pd.Timedelta(minutes=5)
        drop = (local_day == MISSING_DAY) | (
            (local_day == PARTIAL_DAY) & (rng.random(len(raw)) < 0.6)
        )
        raw = raw[~drop]
        replay = raw.sample(frac=0.003, random_state=int(rng.integers(1e9)))
        raw = pd.concat([raw, replay], ignore_index=True).sort_values("call_id")
        self.truth["calls"] = {
            "pbx_migration_date": str(PBX_MIGRATION.date()),
            "timezone_shift_hours_before": -3,
            "missing_day": str(MISSING_DAY.date()),
            "partial_day": str(PARTIAL_DAY.date()),
            "replayed_rows": int(len(replay)),
            "broken_time_order": int(broken.sum()),
        }
        return raw.reset_index(drop=True)

    def run(self, paths: Paths) -> dict:
        sp = self.service_points()
        cust = self.customers(sp)
        veh = self.vehicles(cust)
        so = self.service_orders(cust, veh)
        raw_cust, dup_ids = self.raw_customers(cust, veh)
        calls = self.calls(cust, veh, dup_ids)
        tickets = self.tickets(calls, cust, veh)
        surveys = self.surveys(calls, veh)

        models = pd.DataFrame(
            MODELS,
            columns=[
                "model",
                "brand",
                "segment",
                "is_ev",
                "launch_year",
                "weight",
                "complaint_mult",
                "wmi",
            ],
        )
        out = {
            "models": models[["model", "brand", "segment", "is_ev", "launch_year"]],
            "service_points": sp,
            "customers": raw_cust,
            "vehicles": self.raw_vehicles(veh),
            "service_orders": so,
            "calls": self.raw_calls(calls),
            "tickets": tickets,
            "surveys": surveys,
        }
        for name, df in out.items():
            df.to_parquet(paths.raw / f"{name}.parquet", index=False)

        self.truth.pop("duplicate_customers").to_parquet(
            paths.truth / "duplicate_customers.parquet", index=False
        )
        veh[["vin", "p_post_warranty_visit"]].to_parquet(
            paths.truth / "vehicle_loyalty.parquet", index=False
        )
        self.truth["events"] = {
            "recall_start": str(RECALL_START.date()),
            "price_increase_start": str(PRICE_INCREASE_START.date()),
            "ev_issue_start": str(EV_ISSUE_START.date()),
        }
        self.truth["config"] = asdict(self.cfg)
        (paths.truth / "injected.json").write_text(json.dumps(self.truth, indent=2, default=str))
        return {name: len(df) for name, df in out.items()}


def generate(paths: Paths, cfg: GenConfig | None = None) -> dict:
    """Write the raw extracts to ``paths.raw`` and return row counts."""
    return _Generator(cfg or GenConfig()).run(paths)
