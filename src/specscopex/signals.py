from __future__ import annotations

import numpy as np
import pandas as pd


def _select_representative_price(latest_prices: list[dict]) -> float | None:
    if not latest_prices:
        return None

    df = pd.DataFrame(latest_prices)
    if "price_jpy" not in df:
        return None

    df = df[df["price_jpy"].notnull()]
    if df.empty:
        return None

    df["stock_status"] = df["stock_status"].fillna("")
    in_stock = df[df["stock_status"].str.contains("在庫あり")]
    if not in_stock.empty:
        return float(in_stock["price_jpy"].min())

    return float(df["price_jpy"].min())


def _calc_trend_slope(history_df: pd.DataFrame) -> float | None:
    if history_df.empty or "scraped_at" not in history_df:
        return None

    df = history_df.copy()
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df = df.sort_values("scraped_at")

    cutoff = df["scraped_at"].max() - pd.Timedelta(days=7)
    df_recent = df[df["scraped_at"] >= cutoff]
    if len(df_recent) < 2:
        return None

    x = (df_recent["scraped_at"] - df_recent["scraped_at"].min()).dt.total_seconds() / 86400
    y = df_recent["price_jpy"].astype(float)
    if len(x) < 2:
        return None

    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def compute_signal(latest_prices: list[dict], history_30: list[dict]) -> dict:
    history_df = pd.DataFrame(history_30 or [])
    history_df = history_df[history_df["price_jpy"].notnull()] if not history_df.empty else history_df

    p_now = _select_representative_price(latest_prices or [])
    p_min30 = float(history_df["price_jpy"].min()) if not history_df.empty else None
    p_avg30 = float(history_df["price_jpy"].mean()) if not history_df.empty else None

    data_points = len(history_df)
    data_insufficient = data_points < 5 or any(v is None for v in [p_now, p_min30, p_avg30])

    ratio_min = None if data_insufficient or p_min30 == 0 else (p_now - p_min30) / p_min30  # type: ignore[operator]
    ratio_avg = None if data_insufficient or p_avg30 == 0 else (p_now - p_avg30) / p_avg30  # type: ignore[operator]
    trend7 = _calc_trend_slope(history_df) if not history_df.empty else None

    decision = "check"
    conclusion = "結論: データ不足のため確認が必要"

    if not data_insufficient:
        if ratio_min is not None and ratio_avg is not None and ratio_min <= 0.03 and ratio_avg <= 0.0:
            decision = "buy"
            conclusion = "結論: 直近30日で最安圏なので買い"
        elif ratio_avg is not None and (ratio_avg >= 0.08 or (trend7 is not None and trend7 > 0)):
            decision = "wait"
            conclusion = "結論: 平均より高め・上昇傾向なので待ち"
        elif ratio_avg is not None and abs(ratio_avg) <= 0.05:
            decision = "check"
            conclusion = "結論: 価格は平均付近なので様子見"
        else:
            conclusion = "結論: 価格動向を確認"

    trend_direction = "—"
    if trend7 is not None:
        if trend7 > 0:
            trend_direction = "↑"
        elif trend7 < 0:
            trend_direction = "↓"
        else:
            trend_direction = "→"

    return {
        "decision": decision,
        "status_label": {"buy": "🔵 Buy", "check": "🟡 Check", "wait": "🔴 Wait"}.get(decision, "🟡 Check"),
        "conclusion": conclusion,
        "metrics": {
            "price_now": p_now,
            "price_min30": p_min30,
            "price_avg30": p_avg30,
            "ratio_min": ratio_min,
            "ratio_avg": ratio_avg,
            "trend7": trend7,
            "trend_direction": trend_direction,
            "data_points": data_points,
            "data_insufficient": data_insufficient,
        },
    }
