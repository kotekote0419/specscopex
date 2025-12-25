from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

UNKNOWN_SHOP = "unknown"


@dataclass(frozen=True)
class UrlSeed:
    sku_id: str
    shop: str
    url: str
    title: str | None = None
    is_active: bool = True


def _connect() -> psycopg.Connection:
    load_dotenv()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set. Put it in .env or environment variables.")
    return psycopg.connect(dsn, row_factory=dict_row)


def _load_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("urls"), list):
        return raw["urls"]
    raise ValueError("urls_seed.json must be a list OR {'urls': [...]}")

def _norm_shop(shop: str | None) -> str:
    s = (shop or "").strip().lower()
    return s if s else UNKNOWN_SHOP


def _to_seed(item: dict[str, Any]) -> UrlSeed:
    sku_id = str(item.get("sku_id") or "").strip()
    url = str(item.get("url") or "").strip()
    shop = _norm_shop(item.get("shop"))
    title = item.get("title")
    is_active = item.get("is_active", True)

    if not sku_id or not url:
        raise ValueError(f"Missing required keys: sku_id/url in item={item}")

    if title is not None:
        title = str(title).strip() or None

    return UrlSeed(
        sku_id=sku_id,
        shop=shop,
        url=url,
        title=title,
        is_active=bool(is_active),
    )


def upsert_urls(seeds: Iterable[UrlSeed]) -> int:
    sql = """
    insert into public.product_urls (sku_id, shop, url, title, is_active, created_at, updated_at)
    values (%(sku_id)s, %(shop)s, %(url)s, %(title)s, %(is_active)s, now(), now())
    on conflict (sku_id, shop, url) do update
    set
      title = coalesce(excluded.title, public.product_urls.title),
      is_active = excluded.is_active,
      updated_at = now()
    ;
    """
    rows = []
    for s in seeds:
        rows.append(
            {
                "sku_id": s.sku_id,
                "shop": s.shop,
                "url": s.url,
                "title": s.title,
                "is_active": s.is_active,
            }
        )

    if not rows:
        return 0

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

    return len(rows)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/urls_seed.json", help="Path to urls_seed.json")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    items = _load_json(path)
    seeds = [_to_seed(it) for it in items]
    n = upsert_urls(seeds)
    print(f"OK: upserted {n} product URLs into public.product_urls")


if __name__ == "__main__":
    main()
