from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from ..collectors.price import PriceResult, collect_price
from ..db import (
    ensure_schema,
    list_price_targets,
    upsert_price_snapshot,
    upsert_product_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetUrl:
    sku_id: str
    shop: str | None
    url: str
    product_url_id: Optional[int] = None


def _scraped_date_jst(scraped_at: str) -> str:
    dt = datetime.fromisoformat(scraped_at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    jst = dt.astimezone(timezone(timedelta(hours=9)))
    return jst.date().isoformat()


def enumerate_targets() -> list[TargetUrl]:
    raw_rows = list_price_targets()
    targets = [TargetUrl(**row) for row in raw_rows]
    if not targets:
        logger.warning("no target URLs found in product_aliases")
    else:
        logger.info("found %d target URLs", len(targets))
    return targets


def persist_result(target: TargetUrl, result: PriceResult) -> None:
    product_url_id = target.product_url_id
    if product_url_id is None:
        product_url_id = upsert_product_url(
            sku_id=target.sku_id,
            shop=target.shop or "",
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
        logger.info("processing target", extra={"sku_id": target.sku_id, "shop": target.shop, "url": target.url})
        try:
            result = collect_price(shop=target.shop, url=target.url)
        except Exception:
            logger.exception("failed to collect price", extra={"url": target.url, "shop": target.shop})
            continue

        try:
            persist_result(target, result)
            logger.info(
                "saved price history",
                extra={
                    "sku_id": target.sku_id,
                    "shop": target.shop,
                    "url": target.url,
                    "price_jpy": result.price_jpy,
                    "scraped_at": result.scraped_at,
                },
            )
        except Exception:
            logger.exception("failed to save price history", extra={"url": target.url, "shop": target.shop})
            continue


def main() -> None:
    ensure_schema()
    targets = enumerate_targets()
    collect_for_targets(targets)


if __name__ == "__main__":
    main()
