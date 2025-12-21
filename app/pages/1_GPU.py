from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from specscopex.db import (
    ensure_schema,
    get_latest_prices_by_sku,
    get_price_history,
    list_products,
)


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


products = load_products()
if not products:
    st.warning("プロダクトデータがまだありません。価格収集ジョブ実行後に再度お試しください。")
    st.stop()


options = {
    f"{p['display_name']} ({p['sku_id']})": p["sku_id"]
    for p in products
}
selected_label = st.selectbox("SKU を選択", options.keys())
selected_sku = options[selected_label]

product = next((p for p in products if p["sku_id"] == selected_sku), None)
if product:
    st.subheader(product["display_name"])
else:
    st.subheader(selected_sku)


latest_prices = load_latest_prices(selected_sku)
history_30 = load_price_history(selected_sku, days=30)
history_all = load_price_history(selected_sku, days=None)


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


def render_history(prices: list[dict], title: str) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


render_latest(latest_prices)

col1, col2 = st.columns(2)
with col1:
    render_history(history_30, "直近30日の価格推移")
with col2:
    render_history(history_all, "全期間の価格推移")
