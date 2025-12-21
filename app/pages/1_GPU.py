from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="GPU", page_icon="🖥️", layout="wide")

st.title("GPU（ダミー）")
st.caption("ここに価格推移 / 信号機 / 予測レンジを表示していきます。")

st.markdown(
    """
MVPではまず以下を表示します：

- 🔵🟡🔴 買い時信号機（根拠3行）
- 価格推移チャート（30日〜全期間）
- 予測レンジ（7/14/30日）＋精度（MAPE）
"""
)
