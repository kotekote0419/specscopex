from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import get_settings
from .llm import LLMError, llm_explain_signal
from .utils import json_dumps, now_iso, utc_now_iso


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)  # ensure ./data exists
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    with _connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
            payload_json TEXT NOT NULL,
            suggested_json TEXT,
            final_json TEXT,
            confidence REAL,
            needs_review INTEGER,
            reason_code TEXT,
            note TEXT,
            resolver TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            model_id TEXT,
            prompt_version TEXT,
            schema_version TEXT
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_review_status ON admin_review_queue(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_review_item_type ON admin_review_queue(item_type);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            sku_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            normalized_model TEXT,
            variant TEXT,
            memory_gb INTEGER,
            perf_score REAL,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_model ON products(normalized_model);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS product_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT NOT NULL,
            shop TEXT,
            alias_text TEXT,
            url TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (sku_id) REFERENCES products(sku_id)
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alias_sku ON product_aliases(sku_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alias_url ON product_aliases(url);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT NOT NULL,
            shop TEXT,
            url TEXT,
            price_jpy INTEGER,
            stock_status TEXT,
            title TEXT,
            scraped_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            currency TEXT DEFAULT 'JPY',
            FOREIGN KEY (sku_id) REFERENCES products(sku_id)
        );
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_price_unique ON price_history(sku_id, url, scraped_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_price_sku ON price_history(sku_id);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL,
            model_id TEXT,
            prompt_version TEXT,
            schema_version TEXT,
            input_digest TEXT,
            output_json TEXT,
            confidence REAL,
            needs_review INTEGER,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_task_type ON llm_audits(task_type);")

        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS signal_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT NOT NULL,
            signal TEXT NOT NULL,
            signal_hash TEXT NOT NULL,
            template_text TEXT NOT NULL,
            llm_text TEXT,
            llm_model TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(sku_id, signal_hash)
        );
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_explanations_sku ON signal_explanations(sku_id);"
        )

        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS fx_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            base TEXT NOT NULL,
            quote TEXT NOT NULL,
            rate REAL NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(date, base, quote)
        );
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fx_base_quote ON fx_rates(base, quote);")
        conn.commit()


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
            VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_type,
                payload_json,
                suggested_json,
                confidence,
                1 if needs_review else (0 if needs_review is not None else None),
                created_at,
                model_id,
                prompt_version,
                schema_version,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


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
        clauses.append("status = ?")
        params.append(status)
    if item_type:
        clauses.append("item_type = ?")
        params.append(item_type)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_review_item(item_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM admin_review_queue WHERE id = ?", (item_id,)).fetchone()
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
            SET suggested_json = ?,
                confidence = ?,
                needs_review = ?,
                model_id = ?,
                prompt_version = ?,
                schema_version = ?
            WHERE id = ?
            """,
            (
                suggested_json,
                confidence,
                1 if needs_review else (0 if needs_review is not None else None),
                model_id,
                prompt_version,
                schema_version,
                item_id,
            ),
        )
        conn.commit()


def save_review_draft_final(*, item_id: int, final_obj: Any | None) -> None:
    """
    pending のまま final_json を下書き保存する用途。
    final_obj=None の場合は final_json を null に戻す。
    """
    final_json = json_dumps(final_obj) if final_obj is not None else None
    with _connect() as conn:
        conn.execute(
            "UPDATE admin_review_queue SET final_json = ? WHERE id = ?",
            (final_json, item_id),
        )
        conn.commit()


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
            SET status = ?,
                resolver = ?,
                resolved_at = ?,
                final_json = COALESCE(?, final_json),
                reason_code = COALESCE(?, reason_code),
                note = COALESCE(?, note)
            WHERE id = ?
            """,
            (new_status, resolver, resolved_at, final_json, reason_code, note, item_id),
        )
        conn.commit()


# -------------------------
# LLM audit log
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
) -> int:
    created_at = now_iso()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO llm_audits (
                task_type, model_id, prompt_version, schema_version,
                input_digest, output_json, confidence, needs_review, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_type,
                model_id,
                prompt_version,
                schema_version,
                input_digest,
                output_json,
                confidence,
                1 if needs_review else (0 if needs_review is not None else None),
                created_at,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


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
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sku_id, display_name, normalized_model, variant, memory_gb, perf_score, created_at),
        )
        conn.commit()


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
            VALUES (?, ?, ?, ?, ?)
            """,
            (sku_id, shop, alias_text, url, created_at),
        )
        conn.commit()


def list_products(limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM products ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
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
        row = conn.execute(
            """
            SELECT *
            FROM products
            WHERE normalized_model = ?
              AND COALESCE(variant,'') = COALESCE(?, '')
              AND COALESCE(memory_gb,-1) = COALESCE(?, -1)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (normalized_model, variant, memory_gb),
        ).fetchone()
        return dict(row) if row else None


# -------------------------
# Aliases
# -------------------------
def list_aliases_for_sku(*, sku_id: str, limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM product_aliases
            WHERE sku_id = ?
            ORDER BY created_at DESC
            LIMIT ?
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
                FROM product_aliases
                WHERE sku_id = ?
                  AND url = ?
                ORDER BY created_at DESC
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
                FROM product_aliases
                WHERE sku_id = ?
                  AND COALESCE(shop,'') = COALESCE(?, '')
                  AND COALESCE(alias_text,'') = COALESCE(?, '')
                ORDER BY created_at DESC
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
    Enumerate alias URLs to be scraped.
    Only aliases with non-empty URL are returned.
    """

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT sku_id, shop, url
            FROM product_aliases
            WHERE url IS NOT NULL AND TRIM(url) != ''
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def insert_price_history(
    *,
    sku_id: str,
    shop: str | None,
    url: str,
    price_jpy: int | None,
    stock_status: str | None,
    title: str | None,
    scraped_at: str,
    currency: str = "JPY",
) -> None:
    created_at = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO price_history (
                sku_id, shop, url, price_jpy, stock_status, title, scraped_at, created_at, currency
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sku_id, url, scraped_at) DO UPDATE SET
                price_jpy = excluded.price_jpy,
                stock_status = excluded.stock_status,
                title = COALESCE(excluded.title, price_history.title),
                currency = excluded.currency
            """,
            (sku_id, shop, url, price_jpy, stock_status, title, scraped_at, created_at, currency),
        )
        conn.commit()


def get_latest_prices_by_sku(*, sku_id: str) -> list[dict[str, Any]]:
    """
    Return the latest price rows per (shop, url).
    """

    with _connect() as conn:
        rows = conn.execute(
            """
            WITH latest AS (
                SELECT sku_id, shop, url, MAX(scraped_at) AS max_scraped_at
                FROM price_history
                WHERE sku_id = ?
                GROUP BY shop, url
            )
            SELECT ph.*
            FROM price_history ph
            JOIN latest l
                ON ph.sku_id = l.sku_id
               AND ph.url = l.url
               AND ph.scraped_at = l.max_scraped_at
               AND (
                    (ph.shop = l.shop)
                    OR (ph.shop IS NULL AND l.shop IS NULL)
               )
            WHERE ph.sku_id = ?
            ORDER BY COALESCE(ph.shop, ''), ph.url
            """,
            (sku_id, sku_id),
        ).fetchall()
        return [dict(r) for r in rows]


def get_price_history(*, sku_id: str, days: int | None = None) -> list[dict[str, Any]]:
    clauses = ["sku_id = ?"]
    params: list[Any] = [sku_id]

    if days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(microsecond=0).isoformat()
        clauses.append("scraped_at >= ?")
        params.append(cutoff)

    where_clause = " AND ".join(clauses)

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM price_history WHERE {where_clause} ORDER BY scraped_at",
            params,
        ).fetchall()
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
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, base, quote) DO UPDATE SET
                rate = excluded.rate,
                source = excluded.source
            """,
            rows,
        )
        conn.commit()


def get_fx_rates(
    base: str, quote: str, start_date: str, end_date: str
) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT date, rate
            FROM fx_rates
            WHERE base = ? AND quote = ?
              AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            (base, quote, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


# -------------------------
# Signal explanations
# -------------------------
def _row_to_signal_explanation(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    return data


def get_or_create_explanation(
    *,
    sku_id: str,
    signals: dict[str, Any],
    signal_hash: str,
    template_text: str,
    llm_enabled: bool,
) -> dict[str, Any]:
    created_at = now_iso()

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM signal_explanations WHERE sku_id = ? AND signal_hash = ?",
            (sku_id, signal_hash),
        ).fetchone()

        if row is None:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO signal_explanations (
                    sku_id, signal, signal_hash, template_text, llm_text, llm_model, created_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?)
                """,
                (sku_id, signals.get("signal", ""), signal_hash, template_text, created_at),
            )
            conn.commit()
            if cur.lastrowid:
                row = conn.execute(
                    "SELECT * FROM signal_explanations WHERE id = ?",
                    (int(cur.lastrowid),),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM signal_explanations WHERE sku_id = ? AND signal_hash = ?",
                    (sku_id, signal_hash),
                ).fetchone()
        else:
            if row["template_text"] != template_text:
                conn.execute(
                    """
                    UPDATE signal_explanations
                    SET template_text = ?,
                        llm_text = NULL,
                        llm_model = NULL
                    WHERE id = ?
                    """,
                    (template_text, row["id"]),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM signal_explanations WHERE id = ?",
                    (row["id"],),
                ).fetchone()

    explanation = _row_to_signal_explanation(row) if row else {}

    if llm_enabled and explanation and not explanation.get("llm_text"):
        try:
            llm_text, model_id = llm_explain_signal(
                template_text=template_text,
                signals=signals,
            )
        except LLMError:
            return explanation

        if llm_text:
            explanation["llm_text"] = llm_text
            explanation["llm_model"] = model_id
            with _connect() as conn:
                conn.execute(
                    "UPDATE signal_explanations SET llm_text = ?, llm_model = ? WHERE id = ?",
                    (llm_text, model_id, explanation.get("id")),
                )
                conn.commit()

    return explanation
