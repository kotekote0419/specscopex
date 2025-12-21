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
```

## Run price collection
1. DB スキーマを自動で作成した上で、product_aliases の URL を巡回して価格収集します。
2. 既存の `price_history` に `(sku_id, url, scraped_at)` が同じデータがある場合は上書きします。

```bash
PYTHONPATH=src python -m specscopex.jobs.collect_prices
```

## Launch Streamlit dashboard
価格データと SKU を参照して GPU 価格ダッシュボードを表示します。

```bash
PYTHONPATH=src streamlit run app/Home.py
```
