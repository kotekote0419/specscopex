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

為替重ね表示のオプションをONにすると、Frankfurter（https://api.frankfurter.dev のAPIキー不要）からUSD/JPYを取得して価格推移チャートに第2軸で重ねて表示します。

## 信号機ロジック概要
- 各SKUの最新価格から「在庫あり」を優先して最安の代表値を決定
- 30日履歴で最安・平均・7日傾向（線形回帰の傾き）を算出
- 最安差3%以内かつ平均以下で Buy、平均±5%は Check、平均+8%以上または上昇傾向は Wait

## 買い時根拠の生成（テンプレ + LLM補足）
- 価格シグナルの根拠は必ずテンプレ文章で生成され、数値不足があればその旨を明記します。
- オプションの「AIコメントを表示」トグルをONにすると、テンプレ＋signalsを渡して1〜2文の補足をLLMから取得します。キーが未設定・エラー時は静かにスキップします。
- `signal_explanations` テーブルで `UNIQUE(sku_id, signal_hash)` によりキャッシュされ、同じSKU・同じシグナル条件ではLLMを呼び直しません。

### LLMを使う場合の環境変数
- `OPENAI_API_KEY`: OpenAI APIキー。設定しない場合、テンプレ根拠のみ表示されます。
