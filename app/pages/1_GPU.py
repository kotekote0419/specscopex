from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from specscopex.db import (
    ensure_schema,
    get_fx_rates,
    get_latest_prices_by_sku,
    get_price_history,
    list_products,
    upsert_fx_rates,
)
from specscopex.explain import get_signal_explanation
from specscopex.fx import fetch_usd_jpy_rates
from specscopex.signals import compute_signal
from datetime import date, timedelta


st.set_page_config(page_title="GPU", page_icon="🖥️", layout="wide")
ensure_schema()

st.title("GPU 価格ダッシュボード")
st.caption("SKUごとの最新価格と推移を確認できます。")


@st.cache_data(show_spinner=False)
def load_products() -> list[dict]:
    return list_products(limit=500)


@st.cache_data(show_spinner=False)
def load_latest_prices(sku_id: str) -> list[dict]:
    return get_latest_prices_by_sku(sku_id=sku_id)


@st.cache_data(show_spinner=False)
def load_price_history(sku_id: str, days: int | None = None) -> list[dict]:
    return get_price_history(sku_id=sku_id, days=days)


@st.cache_data(show_spinner=False)
def load_fx_rates(base: str, quote: str, start_date: str, end_date: str) -> list[dict]:
    return get_fx_rates(base=base, quote=quote, start_date=start_date, end_date=end_date)


products = load_products()
if not products:
    st.warning("プロダクトデータがまだありません。価格収集ジョブ実行後に再度お試しください。")
    st.stop()

options = {f"{p['display_name']} ({p['sku_id']})": p["sku_id"] for p in products}
selected_label = st.selectbox("SKU を選択", list(options.keys()))
selected_sku = options[selected_label]

product = next((p for p in products if p["sku_id"] == selected_sku), None)
if product:
    st.subheader(product["display_name"])
else:
    st.subheader(selected_sku)

latest_prices = load_latest_prices(selected_sku)
history_30 = load_price_history(selected_sku, days=30)
history_all = load_price_history(selected_sku, days=None)
signal = compute_signal(latest_prices, history_30)


def _format_price(price: float | int | None) -> str:
    return f"¥{int(price):,}" if price is not None else "—"


def _format_ratio(value: float | None) -> str:
    return f"{value * 100:+.1f}%" if value is not None else "—"


def _build_stock_hint(prices: list[dict]) -> str | None:
    if not prices:
        return None

    statuses = [p.get("stock_status") or "" for p in prices]
    in_stock = [s for s in statuses if "在庫" in s]
    noted = len([s for s in statuses if s.strip()])
    total = len(statuses)
    if noted == 0:
        return None
    return f"在庫表示あり {noted}/{total}件 (在庫あり {len(in_stock)}件)"


def _build_signals_payload(signal_data: dict, prices: list[dict]) -> dict:
    metrics = signal_data.get("metrics", {})
    return {
        "p_now": metrics.get("price_now"),
        "p_min30": metrics.get("price_min30"),
        "p_avg30": metrics.get("price_avg30"),
        "ratio_min": metrics.get("ratio_min"),
        "ratio_avg": metrics.get("ratio_avg"),
        "trend7": metrics.get("trend7"),
        "stock_hint": _build_stock_hint(prices),
        "signal": signal_data.get("decision"),
    }


def _date_range_from_prices(prices: list[dict]) -> tuple[str, str] | None:
    if not prices:
        return None

    df = pd.DataFrame(prices)
    if "scraped_at" not in df:
        return None

    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    start_date = df["scraped_at"].min().date().isoformat()
    end_date = df["scraped_at"].max().date().isoformat()
    return start_date, end_date


def _fetch_and_cache_fx(
    *, base: str, quote: str, start_date: str, end_date: str, failure_flag: dict
) -> list[dict]:
    rates = load_fx_rates(base=base, quote=quote, start_date=start_date, end_date=end_date)
    if rates:
        return rates

    fetched = fetch_usd_jpy_rates(start_date=start_date, end_date=end_date)
    if fetched:
        upsert_fx_rates(base=base, quote=quote, rates_by_date=fetched)
        load_fx_rates.clear()
        return load_fx_rates(base=base, quote=quote, start_date=start_date, end_date=end_date)

    failure_flag["failed"] = True
    return []


def _load_fx_for_prices(
    prices: list[dict], cache: dict[tuple[str, str], list[dict]], failure_flag: dict
) -> list[dict]:
    date_range = _date_range_from_prices(prices)
    if not date_range:
        return []

    start_date, end_date = date_range

    # ★重要：FXは休日/当日未確定で「直近営業日」にズレることがあるのでレンジを広げる
    fx_start = (date.fromisoformat(start_date) - timedelta(days=7)).isoformat()
    fx_end = (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()

    key = (fx_start, fx_end)
    if key in cache:
        return cache[key]

    cache[key] = _fetch_and_cache_fx(
        base="USD", quote="JPY", start_date=fx_start, end_date=fx_end, failure_flag=failure_flag
    )
    return cache[key]


def render_signal_card(signal_data: dict) -> None:
    st.markdown("### 買い時判定（信号機）")
    metrics = signal_data.get("metrics", {})

    card = st.container(border=True)
    with card:
        st.markdown(f"#### {signal_data.get('status_label', '🟡 Check')}")
        st.write(signal_data.get("conclusion", "結論: データ不足"))

        if metrics.get("data_insufficient"):
            st.caption("データ不足：代表値または履歴が不足しています。")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("現在価格（代表値）", _format_price(metrics.get("price_now")))
        col2.metric("30日最安比", _format_ratio(metrics.get("ratio_min")))
        col3.metric("30日平均との差", _format_ratio(metrics.get("ratio_avg")))
        trend_label = metrics.get("trend_direction", "—")
        trend_value = metrics.get("trend7")
        trend_text = f"{trend_label} ({trend_value:.1f})" if trend_value is not None else trend_label
        col4.metric("直近7日のトレンド", trend_text)


def render_explanation_block(explanation: dict, llm_enabled: bool) -> None:
    st.markdown("#### 根拠文章")
    if not explanation:
        st.write("説明を生成できませんでした。")
        return

    st.write(explanation.get("template_text", ""))

    if llm_enabled and explanation.get("llm_text"):
        st.caption("AI補足コメント")
        st.info(explanation["llm_text"], icon="🤖")


def render_latest(prices: list[dict]) -> None:
    st.markdown("### 最新価格（ショップ別）")
    if not prices:
        st.info("まだ価格が登録されていません。価格収集ジョブを実行してください。")
        return

    df = pd.DataFrame(prices)
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    display_cols = ["shop", "price_jpy", "stock_status", "scraped_at", "url", "title"]

    st.dataframe(
        df[display_cols].rename(
            columns={
                "shop": "ショップ",
                "price_jpy": "価格(JPY)",
                "stock_status": "在庫ステータス",
                "scraped_at": "取得時刻",
                "url": "URL",
                "title": "ページタイトル",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_history(
    prices: list[dict], title: str, chart_key: str, fx_rates: list[dict] | None = None
) -> None:
    st.markdown(f"### {title}")
    if not prices:
        st.info("表示できる価格履歴がまだありません。")
        return

    df = pd.DataFrame(prices)
    df = df[df["price_jpy"].notnull()]
    if df.empty:
        st.info("価格データ（数値）が取得できていません。")
        return

    df["scraped_at"] = pd.to_datetime(df["scraped_at"])

    fig = px.line(
        df,
        x="scraped_at",
        y="price_jpy",
        color="shop",
        markers=True,
        hover_data={"url": True, "title": True, "stock_status": True},
        labels={"scraped_at": "取得時刻", "price_jpy": "価格(JPY)", "shop": "ショップ"},
    )
    fig.update_layout(height=420, legend_title_text="ショップ")

    if fx_rates:
        fx_df = pd.DataFrame(fx_rates)
        fx_df["date"] = pd.to_datetime(fx_df["date"])
        fig.add_trace(
            go.Scatter(
                x=fx_df["date"],
                y=fx_df["rate"],
                mode="lines+markers",
                name="USD/JPY",
                yaxis="y2",
                line=dict(color="gray", dash="dash"),
                marker=dict(size=6),
            )
        )
        fig.update_layout(
            yaxis2=dict(title="USD/JPY", overlaying="y", side="right"),
            legend_title_text="凡例",
        )

    # ★重要：keyを必ずユニークにする
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


signals_payload = _build_signals_payload(signal, latest_prices)
fx_cache: dict[tuple[str, str], list[dict]] = {}
fx_failure = {"failed": False}
fx_rates_for_summary: list[dict] | None = None
show_llm_comment = st.toggle(
    "AIコメントを表示",
    value=False,
    help="テンプレ根拠に加えて補足コメントを生成します（同条件はキャッシュされます）。",
    key=f"toggle_ai_comment_{selected_sku}",
)

if show_llm_comment:
    fx_rates_for_summary = _load_fx_for_prices(history_30, fx_cache, fx_failure)

explanation = get_signal_explanation(
    sku_id=selected_sku,
    signals=signals_payload,
    llm_enabled=show_llm_comment,
    fx_rates=fx_rates_for_summary,
)

render_signal_card(signal)
render_explanation_block(explanation, show_llm_comment)

render_latest(latest_prices)

show_fx_overlay = st.checkbox(
    "USD/JPY を重ねて表示",
    value=False,
    help="Frankfurter APIの為替レートを第2軸で表示します（ネットワークに依存）。",
    key=f"toggle_fx_overlay_{selected_sku}",
)

fx_30d: list[dict] | None = None
fx_all: list[dict] | None = None

if show_fx_overlay:
    fx_30d = (
        fx_rates_for_summary
        if fx_rates_for_summary is not None
        else _load_fx_for_prices(history_30, fx_cache, fx_failure)
    )
    fx_all = _load_fx_for_prices(history_all, fx_cache, fx_failure)

col1, col2 = st.columns(2)
with col1:
    render_history(
        history_30,
        "直近30日の価格推移",
        chart_key=f"price_chart_30d_{selected_sku}_{'fx' if show_fx_overlay else 'no_fx'}",
        fx_rates=fx_30d,
    )
with col2:
    render_history(
        history_all,
        "全期間の価格推移",
        chart_key=f"price_chart_all_{selected_sku}_{'fx' if show_fx_overlay else 'no_fx'}",
        fx_rates=fx_all,
    )

if show_fx_overlay and fx_failure.get("failed"):
    st.caption("USD/JPY取得失敗")
