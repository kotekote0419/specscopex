from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

from ..collectors.price import PriceResult, collect_price
from ..db import (
    ensure_schema,
    list_price_targets,
    upsert_price_snapshot,
    upsert_product_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
UNKNOWN_SHOP = "unknown"
HARD_LIMIT_MAX_URLS = 200


@dataclass(frozen=True)
class TargetUrl:
    sku_id: str
    shop: str | None
    url: str
    product_url_id: Optional[int] = None


def _scraped_date_jst(scraped_at: str) -> str:
    """
    Convert scraped_at ISO string to JST date (YYYY-MM-DD).
    If scraped_at is naive (no tzinfo), assume UTC and warn.
    """
    dt = datetime.fromisoformat(scraped_at)
    if dt.tzinfo is None:
        logger.warning("scraped_at is naive; assuming UTC. scraped_at=%s", scraped_at)
        dt = dt.replace(tzinfo=timezone.utc)
    jst_dt = dt.astimezone(JST)
    return jst_dt.date().isoformat()


def _normalize_shop_for_db(shop: str | None) -> str:
    # DB用：NOT NULL前提なのでunknownへ寄せる
    if shop is None or not str(shop).strip():
        return UNKNOWN_SHOP
    return str(shop).strip().lower()


def enumerate_targets() -> list[TargetUrl]:
    raw_rows = list_price_targets()
    targets = [TargetUrl(**row) for row in raw_rows]
    if not targets:
        logger.warning("no target URLs found in product_aliases")
        return targets

    unknown_shop = sum(1 for t in targets if not (t.shop or "").strip())
    logger.info("found %d target URLs (unknown shop=%d)", len(targets), unknown_shop)
    return targets


def persist_result(target: TargetUrl, result: PriceResult) -> None:
    shop_db = _normalize_shop_for_db(target.shop)

    product_url_id = target.product_url_id
    if product_url_id is None:
        product_url_id = upsert_product_url(
            sku_id=target.sku_id,
            shop=shop_db,
            url=target.url,
            title=result.title,
        )

    scraped_date = _scraped_date_jst(result.scraped_at)
    upsert_price_snapshot(
        product_url_id=product_url_id,
        scraped_at=result.scraped_at,
        scraped_date=scraped_date,
        price_jpy=result.price_jpy,
        stock_status=result.stock_status,
    )


def collect_for_targets(targets: Iterable[TargetUrl]) -> None:
    for target in targets:
        shop_db = _normalize_shop_for_db(target.shop)

        logger.info("processing target sku=%s shop=%s url=%s", target.sku_id, shop_db, target.url)

        # collectorには生のshop（None可）を渡す：resolve() がfallbackするため
        try:
            result = collect_price(shop=target.shop, url=target.url)
        except Exception:
            logger.exception("failed to collect price url=%s shop=%s", target.url, shop_db)
            continue

        try:
            persist_result(target, result)
            logger.info(
                "saved price history sku=%s shop=%s price_jpy=%s scraped_at=%s url=%s",
                target.sku_id,
                shop_db,
                result.price_jpy,
                result.scraped_at,
                target.url,
            )
        except Exception:
            logger.exception("failed to save price history url=%s shop=%s", target.url, shop_db)
            continue


def main() -> None:
    ensure_schema()
    targets = enumerate_targets()
    collect_for_targets(targets)


def _emit(logger_fn: Callable[[str], None] | None, message: str) -> None:
    if logger_fn:
        logger_fn(message)
    else:
        logger.info(message)


def _safe_parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def run_collect_prices(
    *,
    sku_id: str | None = None,
    only_active: bool = True,
    limit: int = 1000,
    logger: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    ensure_schema()

    capped_limit = min(int(limit), HARD_LIMIT_MAX_URLS)
    raw_rows = list_price_targets(limit=capped_limit, sku_id=sku_id, only_active=only_active)
    targets = [TargetUrl(**row) for row in raw_rows]

    _emit(logger, f"targets: {len(targets)} (limit={capped_limit})")
    if dry_run:
        _emit(logger, "dry-run enabled: skipping collection")
        return {
            "target_count": len(targets),
            "processed_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "failures": [],
            "latest_scraped_at": None,
            "limit": capped_limit,
        }

    failures: list[dict[str, str]] = []
    success_count = 0
    processed_count = 0
    latest_scraped_at: str | None = None
    latest_dt: datetime | None = None

    for idx, target in enumerate(targets, start=1):
        processed_count += 1
        shop_db = _normalize_shop_for_db(target.shop)
        _emit(logger, f"[{idx}/{len(targets)}] sku={target.sku_id} shop={shop_db} url={target.url}")

        try:
            result = collect_price(shop=target.shop, url=target.url)
        except Exception as exc:
            failures.append(
                {
                    "sku_id": target.sku_id,
                    "shop": shop_db,
                    "url": target.url,
                    "stage": "collect",
                    "error": str(exc),
                }
            )
            _emit(logger, f"failed collect: {target.url} ({exc})")
            continue

        try:
            persist_result(target, result)
            success_count += 1
            _emit(
                logger,
                f"saved price sku={target.sku_id} shop={shop_db} price_jpy={result.price_jpy}",
            )
            parsed_dt = _safe_parse_dt(result.scraped_at)
            if parsed_dt:
                if latest_dt is None or parsed_dt > latest_dt:
                    latest_dt = parsed_dt
                    latest_scraped_at = result.scraped_at
        except Exception as exc:
            failures.append(
                {
                    "sku_id": target.sku_id,
                    "shop": shop_db,
                    "url": target.url,
                    "stage": "persist",
                    "error": str(exc),
                }
            )
            _emit(logger, f"failed persist: {target.url} ({exc})")
            continue

    return {
        "target_count": len(targets),
        "processed_count": processed_count,
        "success_count": success_count,
        "failure_count": len(failures),
        "failures": failures,
        "latest_scraped_at": latest_scraped_at,
        "limit": capped_limit,
    }


if __name__ == "__main__":
    main()
