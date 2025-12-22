from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def summarize_usd_jpy(fx_rates: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Summarize USD/JPY movement for the recent period.

    Args:
        fx_rates: List of dicts with keys {"date": "YYYY-MM-DD", "rate": float}.

    Returns:
        Dict containing:
            - fx_change_30d_pct: Percentage change over the range (first->last).
            - fx_trend_7d: Direction over the last ~7 days (up/down/flat/unknown).
            - fx_last_date / fx_last_rate: Last observed date & rate for transparency.
    """

    default = {
        "fx_change_30d_pct": None,
        "fx_trend_7d": "unknown",
        "fx_last_date": None,
        "fx_last_rate": None,
    }

    if not fx_rates or len(fx_rates) < 2:
        return default

    try:
        parsed = [
            {
                "date": date.fromisoformat(str(item["date"])),
                "rate": float(item["rate"]),
            }
            for item in fx_rates
            if item.get("date") is not None and item.get("rate") is not None
        ]
    except (TypeError, ValueError):
        return default

    if len(parsed) < 2:
        return default

    parsed.sort(key=lambda x: x["date"])
    first, last = parsed[0], parsed[-1]

    change_pct: float | None = None
    if first["rate"]:
        change_pct = (last["rate"] / first["rate"] - 1) * 100

    window_start = last["date"] - timedelta(days=7)
    window_rates = [p for p in parsed if p["date"] >= window_start]

    fx_trend = "unknown"
    if len(window_rates) >= 2:
        ref = window_rates[0]
        diff = last["rate"] - ref["rate"]
        if ref["rate"] == 0:
            fx_trend = "unknown"
        else:
            pct_move = abs(diff) / ref["rate"]
            if pct_move < 0.001:
                fx_trend = "flat"
            elif diff > 0:
                fx_trend = "up"
            else:
                fx_trend = "down"

    return {
        "fx_change_30d_pct": change_pct,
        "fx_trend_7d": fx_trend,
        "fx_last_date": last["date"].isoformat(),
        "fx_last_rate": last["rate"],
    }
