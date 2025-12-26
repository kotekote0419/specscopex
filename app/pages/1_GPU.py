from __future__ import annotations

from datetime import date, timedelta

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
    upsert_forecast_run,
)
from specscopex.explain import get_signal_explanation
from specscopex.forecast import MODEL_NAME as FORECAST_MODEL_NAME, compute_forecast
from specscopex.fx import fetch_usd_jpy_rates
from specscopex.fx_summary import summarize_usd_jpy
from specscopex.llm import LLMError, llm_explain_forecast
from specscopex.signals import compute_signal


st.set_page_config(page_title="GPU", page_icon="🖥️", layout="wide")
ensure_schema()

st.title("GPU 価格ダッシュボード")
st.caption("意思決定に必要な情報を、結論 → 根拠 → 推移 → データの順に整理します。")


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


@st.cache_data(show_spinner=False)
def load_forecast(history: list[dict]) -> dict:
    return compute_forecast(history)


products = load_products()
if not products:
    st.warning("プロダクトデータがまだありません。価格収集ジョブ実行後に再度お試しください。")
    st.stop()

options = {f"{p['display_name']} ({p['sku_id']})": p["sku_id"] for p in products}

with st.sidebar:
    st.header("表示設定", divider=True)
    selected_label = st.selectbox("SKU を選択", list(options.keys()))
    view_days_label = st.radio(
        "表示期間",
        ["7日", "30日", "90日", "全期間"],
        horizontal=True,
    )
    display_mode = st.selectbox(
        "表示モード",
        ["集約（最安）", "集約（平均）", "ショップ別"],
    )
    show_fx_overlay = st.toggle(
        "USD/JPY を重ねる",
        value=False,
        help="Frankfurter APIの為替レートを第2軸で表示します（ネットワークに依存）。",
        key="toggle_fx_overlay",
    )
    show_llm_comment = st.toggle(
        "AIコメントを表示",
        value=False,
        help="テンプレ根拠に加えて補足コメントを生成します（同条件はキャッシュされます）。",
        key="toggle_ai_comment",
    )
    show_forecast_comment = st.toggle(
        "AIで予測コメント（任意）",
        value=False,
        help="予測値とレンジの読み方を1〜2文で補足します（数値はモデル算出固定）。",
        key="toggle_ai_forecast_comment",
    )

selected_sku = options[selected_label]

view_days = {"7日": 7, "30日": 30, "90日": 90, "全期間": None}[view_days_label]

product = next((p for p in products if p["sku_id"] == selected_sku), None)
if product:
    st.subheader(product["display_name"])
else:
    st.subheader(selected_sku)

latest_prices = load_latest_prices(selected_sku)
history_30 = load_price_history(selected_sku, days=30)
history_all = load_price_history(selected_sku, days=None)
history_view = history_all if view_days is None else load_price_history(selected_sku, days=view_days)
signal = compute_signal(latest_prices, history_30)
forecast_result = load_forecast(history_all)


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


def _persist_forecasts(sku_id: str, forecast_data: dict) -> None:
    if not forecast_data.get("ok"):
        return

    as_of = forecast_data.get("as_of")
    features_hash = forecast_data.get("features_hash")
    if not as_of or not features_hash:
        return

    raw = forecast_data.get("forecasts", {}) or {}

    # キーを int に正規化（"7"/"30" でも保存できるようにする）
    forecasts: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            forecasts[int(k)] = v
        except (TypeError, ValueError):
            continue

    for horizon, values in forecasts.items():
        # predicted が無い/None のときは保存しない（0円保存事故を防ぐ）
        pred = (values or {}).get("predicted_price_jpy")
        if pred is None:
            continue

        upsert_forecast_run(
            sku_id=sku_id,
            as_of=as_of,
            horizon_days=int(horizon),
            predicted_price_jpy=float(pred),
            lower_price_jpy=(values or {}).get("lower_price_jpy"),
            upper_price_jpy=(values or {}).get("upper_price_jpy"),
            model_name=forecast_data.get("model_name") or FORECAST_MODEL_NAME,
            features_hash=features_hash,
        )


def render_forecast_section(forecast_data: dict, comment: str | None) -> None:
    st.markdown("### 価格予測（参考）")
    card = st.container(border=True)
    with card:
        if not forecast_data.get("ok"):
            reason = forecast_data.get("reason") or "データ不足"
            st.write(f"予測不可（{reason}）")
            return

        st.caption("統計モデルで算出した参考予測です。")

        raw = forecast_data.get("forecasts", {}) or {}

        # キーを int に正規化（"7"/"30" でもUIで拾えるようにする）
        forecasts: dict[int, dict] = {}
        for k, v in raw.items():
            try:
                forecasts[int(k)] = v
            except (TypeError, ValueError):
                continue

        cols = st.columns(2)
        labels = {7: "7日後", 30: "30日後"}

        for idx, horizon in enumerate((7, 30)):
            data = forecasts.get(horizon)
            col = cols[idx]
            if not data:
                col.write(f"{labels[horizon]}: データなし")
                continue

            col.write(
                f"{labels[horizon]}: "
                f"{_format_price(data.get('predicted_price_jpy'))} "
                f"({_format_price(data.get('lower_price_jpy'))}〜{_format_price(data.get('upper_price_jpy'))})"
            )

        if forecast_data.get("as_of"):
            st.caption(f"基準時刻: {forecast_data['as_of']}")

        if comment:
            st.caption("AI補足コメント（数値はモデル算出済み）")
            st.info(comment, icon="🤖")


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
    display_cols = ["shop", "price_jpy", "stock_status", "scraped_at", "url"]
    df = df[display_cols].sort_values("price_jpy", ascending=True)

    min_row = df.loc[df["price_jpy"].idxmin()] if not df["price_jpy"].isna().all() else None
    if min_row is not None:
        st.success(
            f"最安: {min_row['shop']} / {_format_price(min_row['price_jpy'])}",
            icon="🏷️",
        )

    st.dataframe(
        df.rename(
            columns={
                "shop": "ショップ",
                "price_jpy": "価格(JPY)",
                "stock_status": "在庫ステータス",
                "scraped_at": "取得時刻",
                "url": "URL",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("ショップへのリンク"):
        for _, row in df.iterrows():
            if not row.get("url"):
                continue
            label = f"{row['shop']} ({_format_price(row['price_jpy'])})"
            st.link_button(label, row["url"])


def _prepare_price_frame(prices: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(prices)
    if df.empty:
        return df

    df = df[df["price_jpy"].notnull()]
    if df.empty:
        return df

    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    return df


def render_history(
    prices: list[dict],
    title: str,
    chart_key: str,
    mode: str,
    fx_rates: list[dict] | None = None,
) -> None:
    st.markdown(f"### {title}")
    if not prices:
        st.info("表示できる価格履歴がまだありません。")
        return

    df = _prepare_price_frame(prices)
    if df.empty:
        st.info("価格データ（数値）が取得できていません。")
        return

    if mode in {"集約（最安）", "集約（平均）"}:
        df["date"] = df["scraped_at"].dt.date
        agg_func = "min" if mode == "集約（最安）" else "mean"
        aggregated = df.groupby("date", as_index=False)["price_jpy"].agg(agg_func)
        fig = px.line(
            aggregated,
            x="date",
            y="price_jpy",
            markers=True,
            labels={"date": "日付", "price_jpy": "価格(JPY)"},
        )
        fig.update_layout(height=420, showlegend=False)
    else:
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
_persist_forecasts(selected_sku, forecast_result)
fx_cache: dict[tuple[str, str], list[dict]] = {}
fx_failure = {"failed": False}
fx_rates_for_summary: list[dict] | None = None

if show_llm_comment or show_forecast_comment:
    fx_rates_for_summary = _load_fx_for_prices(history_30, fx_cache, fx_failure)

explanation = get_signal_explanation(
    sku_id=selected_sku,
    signals=signals_payload,
    llm_enabled=show_llm_comment,
    fx_rates=fx_rates_for_summary,
)

forecast_comment: str | None = None
if forecast_result.get("ok") and show_forecast_comment:
    fx_summary_for_comment = summarize_usd_jpy(fx_rates_for_summary)

    # ★ここを追加：forecasts のキーを int に正規化（"7"/"30" 対策）
    raw_fc = forecast_result.get("forecasts", {}) or {}
    norm_fc: dict[int, dict] = {}
    for k, v in raw_fc.items():
        try:
            norm_fc[int(k)] = v
        except (TypeError, ValueError):
            continue

    try:
        forecast_comment, _ = llm_explain_forecast(
            forecasts=norm_fc,  # ★ここを差し替え
            signals=signals_payload,
            fx_summary=fx_summary_for_comment,
        )
    except LLMError:
        forecast_comment = None

latest_df = pd.DataFrame(latest_prices)
latest_min_price = None
latest_updated = None
if not latest_df.empty and "price_jpy" in latest_df:
    latest_min_price = latest_df["price_jpy"].min()
if not latest_df.empty and "scraped_at" in latest_df:
    latest_df["scraped_at"] = pd.to_datetime(latest_df["scraped_at"])
    latest_updated = latest_df["scraped_at"].max()

metric_cols = st.columns(4)
metric_cols[0].metric("今日の最安", _format_price(latest_min_price))
metric_cols[1].metric("30日最安比", _format_ratio(signal.get("metrics", {}).get("ratio_min")))
metric_cols[2].metric("30日平均との差", _format_ratio(signal.get("metrics", {}).get("ratio_avg")))
metric_cols[3].metric(
    "最終更新",
    latest_updated.strftime("%Y-%m-%d %H:%M") if latest_updated is not None else "—",
)

tab_overview, tab_trend, tab_shop, tab_data = st.tabs(["概要", "推移", "ショップ", "データ"])

with tab_overview:
    render_signal_card(signal)
    with st.expander("根拠", expanded=True):
        render_explanation_block(explanation, show_llm_comment)
    with st.expander("予測", expanded=False):
        render_forecast_section(forecast_result, forecast_comment)

with tab_trend:
    fx_view: list[dict] | None = None
    if show_fx_overlay:
        fx_view = _load_fx_for_prices(history_view, fx_cache, fx_failure)

    view_label = view_days_label if view_days is not None else "全期間"
    render_history(
        history_view,
        f"価格推移（{view_label}）",
        chart_key=(
            f"price_chart_view_{selected_sku}_{view_label}_{display_mode}_"
            f"{'fx' if show_fx_overlay else 'no_fx'}"
        ),
        mode=display_mode,
        fx_rates=fx_view,
    )

    if show_fx_overlay and fx_failure.get("failed"):
        st.caption("USD/JPY取得失敗")

with tab_shop:
    render_latest(latest_prices)

with tab_data:
    st.markdown("### 価格履歴（Raw）")
    history_df = pd.DataFrame(history_all)
    if history_df.empty:
        st.info("表示できる価格履歴がまだありません。")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSVをダウンロード",
            data=csv_data,
            file_name=f"{selected_sku}_price_history.csv",
            mime="text/csv",
        )
