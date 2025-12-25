from __future__ import annotations

from typing import Any

import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from .config import get_settings
from .llm import LLMError, llm_explain_signal
from .utils import json_dumps, now_iso, utc_now_iso


def _connect():
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(settings.database_url, row_factory=dict_row)


def ensure_schema() -> None:
    """Supabase側でマイグレーションを適用する前提のため、ここでは no-op。"""
    return None


# -------------------------
# Review queue
# -------------------------
def enqueue_review_item(
    *,
    item_type: str,
    payload_obj: Any,
    suggested_obj: Any | None = None,
    confidence: float | None = None,
    needs_review: bool | None = None,
    model_id: str | None = None,
    prompt_version: str | None = None,
    schema_version: str | None = None,
) -> int:
    created_at = now_iso()
    payload_json = json_dumps(payload_obj)
    suggested_json = json_dumps(suggested_obj) if suggested_obj is not None else None

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO admin_review_queue (
                item_type, status, payload_json, suggested_json,
                confidence, needs_review, created_at,
                model_id, prompt_version, schema_version
            )
            VALUES (%s, 'pending', %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                item_type,
                payload_json,
                suggested_json,
                confidence,
                True if needs_review else (False if needs_review is not None else None),
                created_at,
                model_id,
                prompt_version,
                schema_version,
            ),
        )
        row = cur.fetchone()
        return int(row["id"]) if row else 0


def list_review_items(
    *,
    status: str | None = None,
    item_type: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM admin_review_queue"
    clauses: list[str] = []
    params: list[Any] = []

    if status:
        clauses.append("status = %s")
        params.append(status)
    if item_type:
        clauses.append("item_type = %s")
        params.append(item_type)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY id DESC LIMIT %s"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_review_item(item_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM admin_review_queue WHERE id = %s", (item_id,)).fetchone()
        return dict(row) if row else None


def update_review_suggested(
    *,
    item_id: int,
    suggested_obj: Any,
    confidence: float | None,
    needs_review: bool | None,
    model_id: str | None,
    prompt_version: str | None,
    schema_version: str | None,
) -> None:
    suggested_json = json_dumps(suggested_obj)
    with _connect() as conn:
        conn.execute(
            """
            UPDATE admin_review_queue
            SET suggested_json = %s,
                confidence = %s,
                needs_review = %s,
                model_id = %s,
                prompt_version = %s,
                schema_version = %s
            WHERE id = %s
            """,
            (
                suggested_json,
                confidence,
                True if needs_review else (False if needs_review is not None else None),
                model_id,
                prompt_version,
                schema_version,
                item_id,
            ),
        )


def save_review_draft_final(*, item_id: int, final_obj: Any | None) -> None:
    """
    pending のまま final_json を下書き保存する用途。
    final_obj=None の場合は final_json を null に戻す。
    """
    final_json = json_dumps(final_obj) if final_obj is not None else None
    with _connect() as conn:
        conn.execute(
            "UPDATE admin_review_queue SET final_json = %s WHERE id = %s",
            (final_json, item_id),
        )


def update_review_status(
    *,
    item_id: int,
    new_status: str,
    resolver: str | None,
    final_obj: Any | None = None,
    reason_code: str | None = None,
    note: str | None = None,
) -> None:
    if new_status not in ("pending", "approved", "rejected"):
        raise ValueError("new_status must be one of: pending/approved/rejected")

    resolved_at = now_iso() if new_status in ("approved", "rejected") else None
    final_json = json_dumps(final_obj) if final_obj is not None else None

    with _connect() as conn:
        conn.execute(
            """
            UPDATE admin_review_queue
            SET status = %s,
                resolver = %s,
                resolved_at = %s,
                final_json = COALESCE(%s, final_json),
                reason_code = COALESCE(%s, reason_code),
                note = COALESCE(%s, note)
            WHERE id = %s
            """,
            (new_status, resolver, resolved_at, final_json, reason_code, note, item_id),
        )


# -------------------------
# LLM audits
# -------------------------
def insert_llm_audit(
    *,
    task_type: str,
    model_id: str | None,
    prompt_version: str | None,
    schema_version: str | None,
    input_digest: str,
    output_json: str,
    confidence: float | None,
    needs_review: bool | None,
) -> None:
    created_at = now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO llm_audits (
                task_type, model_id, prompt_version, schema_version,
                input_digest, output_json, confidence, needs_review, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                task_type,
                model_id,
                prompt_version,
                schema_version,
                input_digest,
                output_json,
                confidence,
                True if needs_review else (False if needs_review is not None else None),
                created_at,
            ),
        )


# -------------------------
# Products
# -------------------------
def insert_product(
    *,
    sku_id: str,
    display_name: str,
    normalized_model: str | None,
    variant: str | None,
    memory_gb: int | None,
    perf_score: float | None,
) -> None:
    created_at = now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO products (sku_id, display_name, normalized_model, variant, memory_gb, perf_score, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sku_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                normalized_model = EXCLUDED.normalized_model,
                variant = EXCLUDED.variant,
                memory_gb = EXCLUDED.memory_gb,
                perf_score = EXCLUDED.perf_score
            """,
            (sku_id, display_name, normalized_model, variant, memory_gb, perf_score, created_at),
        )

def upsert_product(
    *,
    sku_id: str,
    display_name: str,
    normalized_model: str | None = None,
    variant: str | None = None,
    memory_gb: int | None = None,
    perf_score: float | None = None,
) -> None:
    """
    URLなしでSKUマスタ(products)を作る/更新するための関数。
    - sku_id が既存なら更新
    - 新規なら追加
    """
    sku_id = (sku_id or "").strip()
    display_name = (display_name or "").strip()

    if not sku_id:
        raise ValueError("sku_id is required")
    if not display_name:
        raise ValueError("display_name is required")

    created_at = now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO products (sku_id, display_name, normalized_model, variant, memory_gb, perf_score, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sku_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                normalized_model = EXCLUDED.normalized_model,
                variant = EXCLUDED.variant,
                memory_gb = EXCLUDED.memory_gb,
                perf_score = EXCLUDED.perf_score
            """,
            (sku_id, display_name, normalized_model, variant, memory_gb, perf_score, created_at),
        )

def insert_alias(
    *,
    sku_id: str,
    shop: str | None,
    alias_text: str | None,
    url: str | None,
) -> None:
    created_at = now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO product_aliases (sku_id, shop, alias_text, url, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (sku_id, shop, alias_text, url) DO NOTHING
            """,
            (sku_id, shop, alias_text, url, created_at),
        )
    if url:
        upsert_product_url(sku_id=sku_id, shop=shop or "", url=url, title=alias_text)


def list_products(limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as conn:
        # Prefer existing products table when available
        try:
            rows = conn.execute(
                "SELECT * FROM products ORDER BY created_at DESC NULLS LAST LIMIT %s",
                (limit,),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        except errors.UndefinedTable:
            pass

        # Fallback: distinct sku_id from product_urls
        rows = conn.execute(
            """
            SELECT sku_id, sku_id AS display_name, MAX(created_at) AS created_at
            FROM product_urls
            WHERE is_active IS TRUE
            GROUP BY sku_id
            ORDER BY MAX(updated_at) DESC NULLS LAST, MAX(created_at) DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def find_product_by_key(
    *,
    normalized_model: str | None,
    variant: str | None,
    memory_gb: int | None,
) -> dict[str, Any] | None:
    """
    重複っぽい product を探す（MVP用の簡易キー）。
    """
    if not normalized_model:
        return None

    with _connect() as conn:
        try:
            row = conn.execute(
                """
                SELECT *
                FROM products
                WHERE normalized_model = %s
                  AND COALESCE(variant,'') = COALESCE(%s, '')
                  AND COALESCE(memory_gb,-1) = COALESCE(%s, -1)
                ORDER BY created_at DESC NULLS LAST
                LIMIT 1
                """,
                (normalized_model, variant, memory_gb),
            ).fetchone()
            return dict(row) if row else None
        except errors.UndefinedTable:
            return None


# -------------------------
# Product URLs (Supabase)
# -------------------------
def upsert_product_url(
    sku_id: str,
    shop: str,
    url: str,
    title: str | None = None,
    is_active: bool = True,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO product_urls (sku_id, shop, url, title, is_active)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (sku_id, shop, url) DO UPDATE
            SET title = COALESCE(EXCLUDED.title, product_urls.title),
                updated_at = NOW(),
                is_active = EXCLUDED.is_active
            RETURNING id
            """,
            (sku_id, shop, url, title, is_active),
        )
        row = cur.fetchone()
        return int(row["id"]) if row else 0

def list_product_urls(
    *,
    sku_id: str | None = None,
    include_inactive: bool = True,
    limit: int = 500,
) -> list[dict[str, Any]]:
    query = """
        SELECT id, sku_id, shop, url, title, is_active, created_at, updated_at
        FROM product_urls
    """
    clauses: list[str] = []
    params: list[Any] = []

    if sku_id:
        clauses.append("sku_id = %s")
        params.append(sku_id)

    if not include_inactive:
        clauses.append("is_active IS TRUE")

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += """
        ORDER BY is_active DESC, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT %s
    """
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def list_product_urls_with_latest_price(
    *,
    sku_id: str | None = None,
    include_inactive: bool = True,
    limit: int = 500,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            pu.id,
            pu.sku_id,
            pu.shop,
            pu.url,
            pu.title,
            pu.is_active,
            pu.created_at,
            pu.updated_at,
            latest.price_jpy AS latest_price_jpy,
            latest.scraped_at AS latest_scraped_at
        FROM product_urls pu
        LEFT JOIN LATERAL (
            SELECT ph.price_jpy, ph.scraped_at
            FROM price_history ph
            WHERE ph.product_url_id = pu.id
            ORDER BY ph.scraped_at DESC
            LIMIT 1
        ) latest ON TRUE
    """
    clauses: list[str] = []
    params: list[Any] = []

    if sku_id:
        clauses.append("pu.sku_id = %s")
        params.append(sku_id)

    if not include_inactive:
        clauses.append("pu.is_active IS TRUE")

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += """
        ORDER BY pu.is_active DESC, pu.updated_at DESC NULLS LAST, pu.created_at DESC NULLS LAST
        LIMIT %s
    """
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def set_product_url_active(*, product_url_id: int, is_active: bool) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE product_urls
            SET is_active = %s,
                updated_at = NOW()
            WHERE id = %s
            """,
            (is_active, product_url_id),
        )


def delete_product_url(*, product_url_id: int) -> None:
    # product_urls -> price_history は ON DELETE CASCADE なので、
    # URLを削除すると、そのURL配下の価格履歴も消えます（意図どおりならOK）
    with _connect() as conn:
        conn.execute("DELETE FROM product_urls WHERE id = %s", (product_url_id,))


def delete_product(*, sku_id: str) -> None:
    with _connect() as conn:
        alias_count = 0
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM product_aliases WHERE sku_id = %s",
                (sku_id,),
            ).fetchone()
            alias_count = int(row["cnt"]) if row else 0
        except errors.UndefinedTable:
            alias_count = 0

        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM product_urls WHERE sku_id = %s",
            (sku_id,),
        ).fetchone()
        url_count = int(row["cnt"]) if row else 0

        effective_alias_count = alias_count if alias_count > 0 else url_count
        if effective_alias_count > 0:
            raise ValueError("alias_count > 0; cannot delete product with aliases or URLs")

        conn.execute("DELETE FROM product_urls WHERE sku_id = %s", (sku_id,))
        conn.execute("DELETE FROM products WHERE sku_id = %s", (sku_id,))


def list_aliases_for_sku(*, sku_id: str, limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as conn:
        try:
            rows = conn.execute(
                """
                SELECT *
                FROM product_aliases
                WHERE sku_id = %s
                ORDER BY created_at DESC NULLS LAST
                LIMIT %s
                """,
                (sku_id, limit),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        except errors.UndefinedTable:
            pass

        rows = conn.execute(
            """
            SELECT id, sku_id, shop, url, title AS alias_text, created_at
            FROM product_urls
            WHERE sku_id = %s
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            LIMIT %s
            """,
            (sku_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def find_alias_duplicate(
    *,
    sku_id: str,
    url: str | None,
    shop: str | None,
    alias_text: str | None,
) -> dict[str, Any] | None:
    """
    alias の重複っぽいものを探す（MVP）。
    - URLがあるならURL一致を優先
    - URLがないなら (sku_id + shop + alias_text) で一致
    """
    with _connect() as conn:
        if url:
            row = conn.execute(
                """
                SELECT *
                FROM product_urls
                WHERE sku_id = %s AND url = %s
                ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
                LIMIT 1
                """,
                (sku_id, url),
            ).fetchone()
            if row:
                return dict(row)

        if alias_text:
            row = conn.execute(
                """
                SELECT *
                FROM product_urls
                WHERE sku_id = %s
                  AND COALESCE(shop,'') = COALESCE(%s, '')
                  AND COALESCE(title,'') = COALESCE(%s, '')
                ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
                LIMIT 1
                """,
                (sku_id, shop, alias_text),
            ).fetchone()
            if row:
                return dict(row)

    return None


# -------------------------
# Price history
# -------------------------
def list_price_targets(limit: int = 1000) -> list[dict[str, Any]]:
    """
    Enumerate URLs to be scraped. Prefers product_urls over legacy aliases.
    Only active URLs are returned.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT sku_id, shop, url, id AS product_url_id
            FROM product_urls
            WHERE is_active IS TRUE
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        if rows:
            return [dict(r) for r in rows]

        try:
            legacy_rows = conn.execute(
                """
                SELECT sku_id, shop, url, NULL::bigint AS product_url_id
                FROM product_aliases
                WHERE url IS NOT NULL AND TRIM(url) != ''
                ORDER BY created_at DESC NULLS LAST
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in legacy_rows]
        except errors.UndefinedTable:
            return []


def upsert_price_snapshot(
    *,
    product_url_id: int,
    scraped_at: str,
    scraped_date: str,
    price_jpy: int | None,
    stock_status: str | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO price_history (
                product_url_id, scraped_at, scraped_date, price_jpy, stock_status
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (product_url_id, scraped_date) DO UPDATE
            SET price_jpy = EXCLUDED.price_jpy,
                stock_status = EXCLUDED.stock_status,
                scraped_at = EXCLUDED.scraped_at
            WHERE (EXCLUDED.price_jpy, EXCLUDED.stock_status)
                  IS DISTINCT FROM (price_history.price_jpy, price_history.stock_status)
            """,
            (product_url_id, scraped_at, scraped_date, price_jpy, stock_status),
        )


def get_latest_prices_by_sku(*, sku_id: str) -> list[dict[str, Any]]:
    """
    Return the latest price rows per product URL.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                pu.id AS product_url_id,
                pu.sku_id,
                pu.shop,
                pu.url,
                pu.title,
                ph.price_jpy,
                ph.stock_status,
                ph.scraped_at,
                ph.scraped_date
            FROM product_urls pu
            JOIN LATERAL (
                SELECT price_jpy, stock_status, scraped_at, scraped_date
                FROM price_history ph
                WHERE ph.product_url_id = pu.id
                ORDER BY ph.scraped_at DESC
                LIMIT 1
            ) ph ON TRUE
            WHERE pu.sku_id = %s AND pu.is_active IS TRUE
            ORDER BY COALESCE(pu.shop, ''), pu.url
            """,
            (sku_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_price_history(*, sku_id: str, days: int | None = None) -> list[dict[str, Any]]:
    clauses = ["pu.sku_id = %s"]
    params: list[Any] = [sku_id]

    if days is not None:
        clauses.append("ph.scraped_at >= NOW() - (%s || ' days')::interval")
        params.append(days)

    where_clause = " AND ".join(clauses)

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT
                pu.id AS product_url_id,
                pu.sku_id,
                pu.shop,
                pu.url,
                pu.title,
                ph.price_jpy,
                ph.stock_status,
                ph.scraped_at,
                ph.scraped_date
            FROM product_urls pu
            JOIN price_history ph ON ph.product_url_id = pu.id
            WHERE {where_clause}
            ORDER BY ph.scraped_at
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]


# -------------------------
# Forecast runs
# -------------------------
def upsert_forecast_run(
    *,
    sku_id: str,
    as_of: str,
    horizon_days: int,
    predicted_price_jpy: float,
    lower_price_jpy: float | None,
    upper_price_jpy: float | None,
    model_name: str,
    features_hash: str,
) -> None:
    created_at = now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO forecast_runs (
                sku_id, as_of, horizon_days, predicted_price_jpy,
                lower_price_jpy, upper_price_jpy, model_name, features_hash, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(sku_id, as_of, horizon_days, model_name, features_hash)
            DO UPDATE SET
                predicted_price_jpy = EXCLUDED.predicted_price_jpy,
                lower_price_jpy = EXCLUDED.lower_price_jpy,
                upper_price_jpy = EXCLUDED.upper_price_jpy,
                created_at = EXCLUDED.created_at
            """,
            (
                sku_id,
                as_of,
                horizon_days,
                predicted_price_jpy,
                lower_price_jpy,
                upper_price_jpy,
                model_name,
                features_hash,
                created_at,
            ),
        )


def get_latest_forecasts(*, sku_id: str, model_name: str, limit: int = 10) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM forecast_runs
            WHERE sku_id = %s AND model_name = %s
            ORDER BY as_of DESC, created_at DESC
            LIMIT %s
            """,
            (sku_id, model_name, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_forecasts_for_period(
    *, sku_id: str, start_date: str, end_date: str, model_name: str | None = None
) -> list[dict]:
    clauses = ["sku_id = %s", "as_of BETWEEN %s AND %s"]
    params: list[Any] = [sku_id, start_date, end_date]

    if model_name:
        clauses.append("model_name = %s")
        params.append(model_name)

    query = "SELECT * FROM forecast_runs WHERE " + " AND ".join(clauses)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# -------------------------
# FX rates
# -------------------------
def upsert_fx_rates(
    base: str, quote: str, rates_by_date: dict[str, float], source: str = "frankfurter"
) -> None:
    created_at = utc_now_iso()
    rows = [
        (date, base, quote, float(rate), source, created_at)
        for date, rate in rates_by_date.items()
    ]

    if not rows:
        return

    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO fx_rates (date, base, quote, rate, source, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(date, base, quote) DO UPDATE SET
                rate = EXCLUDED.rate,
                source = EXCLUDED.source
            """,
            rows,
        )


def get_fx_rates(
    base: str, quote: str, start_date: str, end_date: str
) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT date, rate
            FROM fx_rates
            WHERE base = %s AND quote = %s
              AND date BETWEEN %s AND %s
            ORDER BY date
            """,
            (base, quote, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


# -------------------------
# Signal explanations
# -------------------------
def _row_to_signal_explanation(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def get_or_create_explanation(
    *,
    sku_id: str,
    signals: dict[str, Any],
    signal_hash: str,
    template_text: str,
    llm_enabled: bool,
    fx_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = now_iso()

    with _connect() as conn:
        try:
            row = conn.execute(
                "SELECT * FROM signal_explanations WHERE sku_id = %s AND signal_hash = %s",
                (sku_id, signal_hash),
            ).fetchone()
        except errors.UndefinedTable:
            row = None

        if row is None:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO signal_explanations (
                        sku_id, signal, signal_hash, template_text, llm_text, llm_model, created_at
                    ) VALUES (%s, %s, %s, %s, NULL, NULL, %s)
                    ON CONFLICT (sku_id, signal_hash) DO NOTHING
                    RETURNING *
                    """,
                    (sku_id, signals.get("signal", ""), signal_hash, template_text, created_at),
                )
                row = cur.fetchone()
            except errors.UndefinedTable:
                row = None

            if row is None:
                try:
                    row = conn.execute(
                        "SELECT * FROM signal_explanations WHERE sku_id = %s AND signal_hash = %s",
                        (sku_id, signal_hash),
                    ).fetchone()
                except errors.UndefinedTable:
                    row = None
        else:
            if row.get("template_text") != template_text:
                conn.execute(
                    """
                    UPDATE signal_explanations
                    SET template_text = %s,
                        llm_text = NULL,
                        llm_model = NULL
                    WHERE id = %s
                    """,
                    (template_text, row.get("id")),
                )
                try:
                    row = conn.execute(
                        "SELECT * FROM signal_explanations WHERE id = %s",
                        (row.get("id"),),
                    ).fetchone()
                except errors.UndefinedTable:
                    row = None

    explanation = _row_to_signal_explanation(row) if row else {}

    if llm_enabled and explanation and not explanation.get("llm_text"):
        try:
            llm_text, model_id = llm_explain_signal(
                template_text=template_text,
                signals=signals,
                fx_summary=fx_summary,
            )
        except LLMError:
            return explanation

        if llm_text:
            explanation["llm_text"] = llm_text
            explanation["llm_model"] = model_id
            with _connect() as conn:
                try:
                    conn.execute(
                        "UPDATE signal_explanations SET llm_text = %s, llm_model = %s WHERE id = %s",
                        (llm_text, model_id, explanation.get("id")),
                    )
                except errors.UndefinedTable:
                    pass

    return explanation


__all__ = [
    "ensure_schema",
    "enqueue_review_item",
    "list_review_items",
    "get_review_item",
    "update_review_suggested",
    "save_review_draft_final",
    "update_review_status",
    "insert_llm_audit",
    "insert_product",
    "upsert_product",
    "insert_alias",
    "list_products",
    "find_product_by_key",
    "list_aliases_for_sku",
    "find_alias_duplicate",
    "list_product_urls",
    "list_product_urls_with_latest_price",
    "set_product_url_active",
    "delete_product_url",
    "delete_product",
    "list_price_targets",
    "upsert_product_url",
    "upsert_price_snapshot",
    "get_latest_prices_by_sku",
    "get_price_history",
    "upsert_forecast_run",
    "get_latest_forecasts",
    "get_forecasts_for_period",
    "upsert_fx_rates",
    "get_fx_rates",
    "get_or_create_explanation",
]
