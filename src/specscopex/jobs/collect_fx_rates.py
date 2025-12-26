from __future__ import annotations

import logging
from datetime import date, timedelta

from ..db import ensure_schema, upsert_fx_rates
from ..fx import fetch_usd_jpy_rates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 60


def collect_usd_jpy() -> None:
    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    logger.info("fetching USD/JPY rates: %s..%s", start_date, end_date)

    try:
        rates = fetch_usd_jpy_rates(start_date=start_date.isoformat(), end_date=end_date.isoformat())
    except Exception:
        logger.exception("failed to fetch USD/JPY rates")
        return

    if not rates:
        logger.warning("no USD/JPY rates fetched")
        return

    try:
        upsert_fx_rates(base="USD", quote="JPY", rates_by_date=rates)
        logger.info("upserted %d USD/JPY rates", len(rates))
    except Exception:
        logger.exception("failed to upsert USD/JPY rates")


def main() -> None:
    ensure_schema()
    collect_usd_jpy()


if __name__ == "__main__":
    main()
