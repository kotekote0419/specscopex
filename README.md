# SpecScopeX — PCパーツ買い時ナビ

Streamlit + Supabase(PostgreSQL) + OpenAI Structured Outputs（Pydantic 4スキーマ）で
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

### Database (Supabase/PostgreSQL)
- Supabase プロジェクトを作成し、環境変数 `DATABASE_URL` に接続文字列を設定してください。
- Supabase の SQL Editor で `migrations/001_supabase_init.sql` を実行し、テーブルを作成してください。

## Run price collection
1. Supabase 上の `product_urls` から URL を巡回して価格収集します（legacy の `product_aliases` はフォールバックのみ）。
2. `product_url_id × scraped_date(JST)` で 1 日 1 件に upsert されます。

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
- AIコメントはUSD/JPY要約（30日変化率・直近7日方向）を参考にする場合がありますが、外部要因としての可能性に留め、因果は断定しません。
- `signal_explanations` テーブルで `UNIQUE(sku_id, signal_hash)` によりキャッシュされ、同じSKU・同じシグナル条件ではLLMを呼び直しません。

### LLMを使う場合の環境変数
- `OPENAI_API_KEY`: OpenAI APIキー。設定しない場合、テンプレ根拠のみ表示されます。

## 価格予測MVP（7日後/30日後）
- 日次代表価格（ショップ別最安の中央値）から線形トレンドで7日後/30日後の予測値とレンジ（残差σベース）を決定論で算出します。
- 予測結果は `forecast_runs` テーブルに保存され、後から答え合わせに使えるようになっています（同条件では重複保存なし）。
- Streamlitでは信号機カード付近に予測を表示し、データ不足時は「予測不可」を明示します。AI補足コメントはトグルON時のみ呼び出し、数値はモデル算出結果をそのまま用います。
