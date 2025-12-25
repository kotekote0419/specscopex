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


@dataclass(frozen=True)
class ProductSeed:
    sku_id: str
    display_name: str
    normalized_model: str | None = None
    variant: str | None = None
    memory_gb: int | None = None
    perf_score: float | None = None
    created_at: str | None = None  # ISO8601 (optional)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Accept either:
    #  - list[dict]
    #  - {"products": [...]}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("products"), list):
        items = raw["products"]
    else:
        raise ValueError("products_seed.json must be a list OR {'products': [...]}")

    out: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"Invalid item at index {i}: not an object")
        out.append(it)
    return out


def _to_seed(item: dict[str, Any]) -> ProductSeed:
    sku_id = str(item.get("sku_id") or item.get("sku") or "").strip()
    display_name = str(item.get("display_name") or item.get("name") or "").strip()
    if not sku_id or not display_name:
        raise ValueError(f"Missing required keys: sku_id/display_name in item={item}")

    normalized_model = item.get("normalized_model")
    variant = item.get("variant")
    memory_gb = item.get("memory_gb")
    perf_score = item.get("perf_score")
    created_at = item.get("created_at")

    if normalized_model is not None:
        normalized_model = str(normalized_model).strip() or None
    if variant is not None:
        variant = str(variant).strip() or None
    if memory_gb is not None:
        memory_gb = int(memory_gb)
    if perf_score is not None:
        perf_score = float(perf_score)
    if created_at is not None:
        created_at = str(created_at).strip() or None

    return ProductSeed(
        sku_id=sku_id,
        display_name=display_name,
        normalized_model=normalized_model,
        variant=variant,
        memory_gb=memory_gb,
        perf_score=perf_score,
        created_at=created_at,
    )


def _connect() -> psycopg.Connection:
    # Load .env from current working directory (project root assumed)
    load_dotenv()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set. Put it in .env (project root) or environment variables.")
    return psycopg.connect(dsn, row_factory=dict_row)


def upsert_products(seeds: Iterable[ProductSeed]) -> int:
    sql = """
    insert into public.products
      (sku_id, display_name, normalized_model, variant, memory_gb, perf_score, created_at)
    values
      (%(sku_id)s, %(display_name)s, %(normalized_model)s, %(variant)s, %(memory_gb)s, %(perf_score)s, %(created_at)s)
    on conflict (sku_id) do update
    set
      display_name = excluded.display_name,
      normalized_model = excluded.normalized_model,
      variant = excluded.variant,
      memory_gb = excluded.memory_gb,
      perf_score = excluded.perf_score
    ;
    """
    rows = []
    for s in seeds:
        rows.append(
            {
                "sku_id": s.sku_id,
                "display_name": s.display_name,
                "normalized_model": s.normalized_model,
                "variant": s.variant,
                "memory_gb": s.memory_gb,
                "perf_score": s.perf_score,
                "created_at": s.created_at or _now_iso(),
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
    p.add_argument("--path", default="data/products_seed.json", help="Path to products_seed.json")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    items = _load_json(path)
    seeds = [_to_seed(it) for it in items]
    n = upsert_products(seeds)
    print(f"OK: upserted {n} products into public.products")


if __name__ == "__main__":
    main()
