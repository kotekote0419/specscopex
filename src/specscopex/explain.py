from __future__ import annotations

import hashlib
import json
from typing import Any

from .db import get_or_create_explanation


def _format_price(value: float | int | None) -> str:
    if value is None:
        return "データ不足"
    return f"¥{int(value):,}"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "データ不足"
    return f"{value:+.1%}"


def _format_trend(trend: float | None) -> tuple[str, str]:
    if trend is None:
        return "データ不足", "―"
    if trend > 0:
        return "上昇", "↑"
    if trend < 0:
        return "下降", "↓"
    return "横ばい", "→"


def render_signal_template(signals: dict[str, Any]) -> str:
    """Generate the mandatory template-based explanation text.

    Parameters
    ----------
    signals: dict
        Expected keys: p_now, p_min30, p_avg30, ratio_min, ratio_avg, trend7,
        stock_hint, signal.
    """

    p_now = signals.get("p_now")
    p_min30 = signals.get("p_min30")
    p_avg30 = signals.get("p_avg30")
    ratio_min = signals.get("ratio_min")
    ratio_avg = signals.get("ratio_avg")
    trend7 = signals.get("trend7")
    stock_hint = signals.get("stock_hint")

    trend_word, trend_arrow = _format_trend(trend7)
    signal_label = {
        "buy": "買い時",  # best timing
        "wait": "待ち",  # hold off
        "check": "要確認",  # need more info
    }.get(signals.get("signal"), "要確認")

    price_sentence = f"現在価格は{_format_price(p_now)}。"

    if p_min30 is None and p_avg30 is None:
        comparison_sentence = "直近30日最安・平均との差はデータ不足で比較不可。"
    else:
        comparison_sentence = (
            f"直近30日最安({_format_price(p_min30)})比 {_format_ratio(ratio_min)}、"
            f"30日平均({_format_price(p_avg30)})比 {_format_ratio(ratio_avg)}。"
        )

    trend_sentence = f"直近7日は{trend_arrow}{trend_word}傾向。"

    stock_sentence = f"在庫傾向: {stock_hint}。" if stock_hint else ""

    conclusion_sentence = f"結論：{signal_label}。"

    parts = [price_sentence, comparison_sentence, trend_sentence]
    if stock_sentence:
        parts.append(stock_sentence)
    parts.append(conclusion_sentence)
    return " ".join(parts)


def _stable_hash_payload(signals: dict[str, Any]) -> str:
    def _round(v: Any) -> Any:
        if isinstance(v, (int, float)):
            return round(float(v), 6)
        return v

    keys = [
        "p_now",
        "p_min30",
        "p_avg30",
        "ratio_min",
        "ratio_avg",
        "trend7",
        "signal",
    ]
    payload = {k: _round(signals.get(k)) for k in keys}
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def get_signal_explanation(
    *, sku_id: str, signals: dict[str, Any], llm_enabled: bool
) -> dict[str, Any]:
    template_text = render_signal_template(signals)
    signal_hash = _stable_hash_payload(signals)
    explanation = get_or_create_explanation(
        sku_id=sku_id,
        signals=signals,
        signal_hash=signal_hash,
        template_text=template_text,
        llm_enabled=llm_enabled,
    )
    return explanation
