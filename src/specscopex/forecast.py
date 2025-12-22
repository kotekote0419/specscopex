from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any

MODEL_NAME = "ma_trend_v1"


def _parse_iso_datetime(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _scraped_to_jst_date(scraped_at: str) -> date | None:
    dt = _parse_iso_datetime(scraped_at)
    if dt is None:
        return None
    jst = dt.astimezone(timezone(timedelta(hours=9)))
    return jst.date()


def _aggregate_daily_representative_prices(price_history: list[dict[str, Any]]):
    daily_shop_min: dict[date, dict[str, float]] = {}

    for row in price_history:
        price = row.get("price_jpy")
        if price is None:
            continue

        scraped_at = row.get("scraped_at")
        if not scraped_at:
            continue

        day = _scraped_to_jst_date(str(scraped_at))
        if day is None:
            continue

        shop = (row.get("shop") or "").strip() or "unknown"
        price_f = float(price)
        shops = daily_shop_min.setdefault(day, {})
        if shop in shops:
            shops[shop] = min(shops[shop], price_f)
        else:
            shops[shop] = price_f

    if not daily_shop_min:
        return []

    daily_rep: list[tuple[date, float]] = []
    for day in sorted(daily_shop_min.keys()):
        prices = list(daily_shop_min[day].values())
        if not prices:
            continue
        if len(prices) == 1:
            rep_price = prices[0]
        else:
            rep_price = float(statistics.median(prices))
        daily_rep.append((day, rep_price))

    return daily_rep


def _linear_trend(daily_rep: list[tuple[date, float]]):
    n = len(daily_rep)
    if n < 2:
        return None, None, None

    x = list(range(n))
    y = [p[1] for p in daily_rep]

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    denom = sum((xi - x_mean) ** 2 for xi in x)
    if denom == 0:
        return None, None, None

    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denom
    intercept = y_mean - slope * x_mean

    fitted = [intercept + slope * xi for xi in x]
    residuals = [yi - fi for yi, fi in zip(y, fitted)]
    return slope, intercept, residuals


def _residual_sigma(residuals: list[float]) -> float:
    if not residuals:
        return 0.0

    window = residuals[-30:]
    if len(window) == 1:
        return 0.0

    mean_r = sum(window) / len(window)
    variance = sum((r - mean_r) ** 2 for r in window) / len(window)
    return math.sqrt(variance)


def _stable_features_hash(payload: dict[str, Any]) -> str:
    rounded = {}
    for key, value in payload.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        elif isinstance(value, list):
            rounded[key] = [round(v, 2) if isinstance(v, (int, float)) else v for v in value]
        else:
            rounded[key] = value

    features_json = json.dumps(rounded, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(features_json.encode("utf-8")).hexdigest()


def compute_forecast(price_history: list[dict[str, Any]]) -> dict[str, Any]:
    daily_rep = _aggregate_daily_representative_prices(price_history)
    if len(daily_rep) < 10:
        return {
            "ok": False,
            "reason": "日次データ不足（10日未満）",
            "as_of": None,
            "model_name": MODEL_NAME,
            "features_hash": None,
            "forecasts": {},
        }

    slope, intercept, residuals = _linear_trend(daily_rep)
    if slope is None or intercept is None or residuals is None:
        return {
            "ok": False,
            "reason": "トレンド計算に失敗",
            "as_of": None,
            "model_name": MODEL_NAME,
            "features_hash": None,
            "forecasts": {},
        }

    sigma = _residual_sigma(residuals)
    last_price = daily_rep[-1][1]

    def _build_forecast(horizon: int) -> dict[str, float]:
        pred = last_price + slope * horizon
        pred_rounded = float(round(pred))
        lower = max(0.0, pred_rounded - sigma)
        upper = pred_rounded + sigma
        return {
            "predicted_price_jpy": pred_rounded,
            "lower_price_jpy": float(round(lower)),
            "upper_price_jpy": float(round(upper)),
        }

    forecasts = {h: _build_forecast(h) for h in (7, 30)}

    parsed_times = [
        _parse_iso_datetime(str(item["scraped_at"]))
        for item in price_history
        if item.get("scraped_at")
    ]
    parsed_times = [dt for dt in parsed_times if dt is not None]
    as_of_dt = max(parsed_times) if parsed_times else datetime.now(timezone.utc)
    as_of = as_of_dt.replace(microsecond=0).isoformat()

    features = {
        "model": MODEL_NAME,
        "as_of": as_of,
        "days": len(daily_rep),
        "start": daily_rep[0][0].isoformat(),
        "end": daily_rep[-1][0].isoformat(),
        "last_price": last_price,
        "slope": slope,
        "sigma": sigma,
        "recent_rep_prices": [p for _, p in daily_rep[-7:]],
    }

    features_hash = _stable_features_hash(features)

    return {
        "ok": True,
        "reason": "",
        "as_of": as_of,
        "model_name": MODEL_NAME,
        "features_hash": features_hash,
        "forecasts": forecasts,
    }

