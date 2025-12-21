# SpecScopeX — PCパーツ買い時ナビ

Streamlit + SQLite + OpenAI Structured Outputs（Pydantic 4スキーマ）で
- Adminレビューキュー（pending/approved/rejected）
- URL貼るだけ追加 → HTML取得 → LLM監査 → sku_candidate → Approveでproducts登録
まで動く最小構成です。

## Requirements
- Python 3.11

## Setup
```bash
cd specscopex
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt --upgrade
