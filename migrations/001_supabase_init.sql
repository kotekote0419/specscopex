-- Product URLs (Supabase / Postgres)
CREATE TABLE IF NOT EXISTS product_urls (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sku_id text NOT NULL,
    shop text NOT NULL,
    url text NOT NULL,
    title text NULL,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE (sku_id, shop, url)
);

CREATE INDEX IF NOT EXISTS idx_product_urls_sku_shop_active
    ON product_urls (sku_id, shop, is_active);

-- Price history (1日1件、Python側で scraped_date を付与)
CREATE TABLE IF NOT EXISTS price_history (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_url_id bigint NOT NULL REFERENCES product_urls(id) ON DELETE CASCADE,
    scraped_at timestamptz NOT NULL,
    scraped_date date NOT NULL,
    price_jpy integer NULL,
    stock_status text NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE (product_url_id, scraped_date),
    CHECK (price_jpy IS NULL OR price_jpy >= 0)
);

CREATE INDEX IF NOT EXISTS idx_price_history_product_date
    ON price_history (product_url_id, scraped_date);
CREATE INDEX IF NOT EXISTS idx_price_history_scraped_at
    ON price_history (scraped_at);

-- Admin review queue
CREATE TABLE IF NOT EXISTS admin_review_queue (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    item_type text NOT NULL,
    status text NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    payload_json jsonb NOT NULL,
    suggested_json jsonb,
    final_json jsonb,
    confidence double precision,
    needs_review boolean,
    reason_code text,
    note text,
    resolver text,
    created_at timestamptz NOT NULL,
    resolved_at timestamptz,
    model_id text,
    prompt_version text,
    schema_version text
);

CREATE INDEX IF NOT EXISTS idx_review_status ON admin_review_queue(status);
CREATE INDEX IF NOT EXISTS idx_review_item_type ON admin_review_queue(item_type);

-- Products
CREATE TABLE IF NOT EXISTS products (
    sku_id text PRIMARY KEY,
    display_name text NOT NULL,
    normalized_model text,
    variant text,
    memory_gb integer,
    perf_score double precision,
    created_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_model ON products(normalized_model);

-- Product aliases (legacy helper)
CREATE TABLE IF NOT EXISTS product_aliases (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sku_id text NOT NULL,
    shop text,
    alias_text text,
    url text,
    created_at timestamptz NOT NULL,
    UNIQUE (sku_id, shop, alias_text, url)
);
CREATE INDEX IF NOT EXISTS idx_alias_sku ON product_aliases(sku_id);
CREATE INDEX IF NOT EXISTS idx_alias_url ON product_aliases(url);

-- LLM audits
CREATE TABLE IF NOT EXISTS llm_audits (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_type text NOT NULL,
    model_id text,
    prompt_version text,
    schema_version text,
    input_digest text,
    output_json jsonb,
    confidence double precision,
    needs_review boolean,
    created_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_task_type ON llm_audits(task_type);

-- Signal explanations
CREATE TABLE IF NOT EXISTS signal_explanations (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sku_id text NOT NULL,
    signal text NOT NULL,
    signal_hash text NOT NULL,
    template_text text NOT NULL,
    llm_text text,
    llm_model text,
    created_at timestamptz NOT NULL,
    UNIQUE (sku_id, signal_hash)
);
CREATE INDEX IF NOT EXISTS idx_signal_explanations_sku ON signal_explanations(sku_id);

-- FX rates
CREATE TABLE IF NOT EXISTS fx_rates (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    date date NOT NULL,
    base text NOT NULL,
    quote text NOT NULL,
    rate double precision NOT NULL,
    source text NOT NULL,
    created_at timestamptz NOT NULL,
    UNIQUE (date, base, quote)
);
CREATE INDEX IF NOT EXISTS idx_fx_base_quote ON fx_rates(base, quote);

-- Forecast runs
CREATE TABLE IF NOT EXISTS forecast_runs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sku_id text NOT NULL,
    as_of timestamptz NOT NULL,
    horizon_days integer NOT NULL,
    predicted_price_jpy double precision NOT NULL,
    lower_price_jpy double precision NULL,
    upper_price_jpy double precision NULL,
    model_name text NOT NULL,
    features_hash text NOT NULL,
    created_at timestamptz NOT NULL,
    UNIQUE (sku_id, as_of, horizon_days, model_name, features_hash)
);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_sku ON forecast_runs(sku_id);
