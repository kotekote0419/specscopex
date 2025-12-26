from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st
from specscopex.db import ensure_schema

st.set_page_config(
    page_title="SpecScopeX — PCパーツ買い時ナビ",
    page_icon="🧭",
    layout="wide",
)

ensure_schema()

st.title("SpecScopeX 🧭")
st.caption("PCパーツ買い時ナビ")

st.markdown(
    """
**買い時、数字で見える。予測も答え合わせも。**

- まずは **GPU** からスタート（CPU/SSDへ拡張予定）
- 管理者ページで **URL貼るだけ追加** → **LLM監査** → **レビュー承認** を回す
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("監視SKU数", "—")
with col2:
    st.metric("今日の観測件数", "—")
with col3:
    st.metric("買いシグナル", "—")

st.info("次のステップ：Adminページで URL貼るだけ追加 → Approve → products登録 を試してください。")
