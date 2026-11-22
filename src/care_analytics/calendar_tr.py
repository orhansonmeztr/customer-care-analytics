"""Turkish public holiday calendar.

Religious holidays follow the lunar calendar and move about 11 days earlier every year,
so "same day last year" baselines break around them. The forecasting features use this
calendar instead.
"""

from __future__ import annotations

import pandas as pd

FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "Yılbaşı",
    (4, 23): "Ulusal Egemenlik ve Çocuk Bayramı",
    (5, 1): "Emek ve Dayanışma Günü",
    (5, 19): "Atatürk'ü Anma, Gençlik ve Spor Bayramı",
    (7, 15): "Demokrasi ve Millî Birlik Günü",
    (8, 30): "Zafer Bayramı",
    (10, 29): "Cumhuriyet Bayramı",
}

# First day of each religious holiday. Ramazan Bayramı lasts 3 days, Kurban Bayramı 4.
RAMAZAN_START = {2023: "2023-04-21", 2024: "2024-04-10", 2025: "2025-03-30", 2026: "2026-03-20"}
KURBAN_START = {2023: "2023-06-28", 2024: "2024-06-16", 2025: "2025-06-06", 2026: "2026-05-27"}


def holiday_calendar(start: str, end: str) -> pd.DataFrame:
    """One row per day with holiday flags.

    Columns: date, holiday_name, holiday_kind ("none" / "national" / "religious"),
    is_holiday, is_arife (eve of a religious holiday, a half working day) and
    is_post_holiday (first working day after a religious holiday, when backlog calls arrive).
    """
    days = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": days})
    cal["holiday_name"] = None
    cal["holiday_kind"] = "none"

    by_date = cal.set_index("date")
    for (month, day), name in FIXED_HOLIDAYS.items():
        mask = (by_date.index.month == month) & (by_date.index.day == day)
        by_date.loc[mask, ["holiday_name", "holiday_kind"]] = [name, "national"]

    arife, post = set(), set()
    for starts, length, name in (
        (RAMAZAN_START, 3, "Ramazan Bayramı"),
        (KURBAN_START, 4, "Kurban Bayramı"),
    ):
        for first in starts.values():
            first = pd.Timestamp(first)
            for i in range(length):
                d = first + pd.Timedelta(days=i)
                if d in by_date.index:
                    by_date.loc[d, ["holiday_name", "holiday_kind"]] = [name, "religious"]
            arife.add(first - pd.Timedelta(days=1))
            post.add(first + pd.Timedelta(days=length))

    # the backlog lands on the first working day: skip weekends and holidays after the bayram
    def next_working_day(d: pd.Timestamp) -> pd.Timestamp:
        while d.day_of_week >= 5 or (
            d in by_date.index and by_date.loc[d, "holiday_kind"] != "none"
        ):
            d += pd.Timedelta(days=1)
        return d

    post = {next_working_day(d) for d in post}

    cal = by_date.reset_index()
    cal["is_holiday"] = cal["holiday_kind"] != "none"
    cal["is_arife"] = cal["date"].isin(arife) & ~cal["is_holiday"]
    cal["is_post_holiday"] = cal["date"].isin(post) & ~cal["is_holiday"]
    return cal
