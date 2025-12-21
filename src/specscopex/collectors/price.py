from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import requests
from bs4 import BeautifulSoup

from ..utils import utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceResult:
    price_jpy: Optional[int]
    stock_status: str
    scraped_at: str
    currency: str = "JPY"
    title: Optional[str] = None


class BaseCollector:
    name: str = "base"

    def collect(self, url: str) -> PriceResult:  # pragma: no cover - interface
        raise NotImplementedError


class GenericHtmlCollector(BaseCollector):
    name = "generic"

    def __init__(self, *, timeout_sec: int = 20) -> None:
        self.timeout_sec = timeout_sec

    def _extract_price(self, text: str) -> Optional[int]:
        cleaned = text.replace(",", "")
        candidates = re.findall(r"(?:¥|\\u00a5|円)?\s*([0-9]{3,8})", cleaned)
        prices: list[int] = []
        for candidate in candidates:
            try:
                value = int(candidate)
            except ValueError:
                continue
            if 100 <= value <= 2_000_000:
                prices.append(value)
        if not prices:
            return None
        # Use the smallest plausible value as a conservative estimate.
        return min(prices)

    def _extract_stock_status(self, text: str) -> str:
        lower_text = text.lower()
        if any(keyword in lower_text for keyword in ["sold out", "out of stock", "在庫なし", "品切", "欠品"]):
            return "out_of_stock"
        if any(keyword in lower_text for keyword in ["在庫あり", "即納", "in stock"]):
            return "in_stock"
        return "unknown"

    def collect(self, url: str) -> PriceResult:
        headers = {
            "User-Agent": "SpecScopeXPriceBot/0.1 (+https://example.local) Python requests",
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        }
        resp = requests.get(url, headers=headers, timeout=self.timeout_sec)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        page_text = soup.get_text(" ", strip=True)
        price_value = self._extract_price(page_text)
        stock_status = self._extract_stock_status(page_text)
        title = soup.title.get_text(" ", strip=True) if soup.title else None

        scraped_at = utc_now_iso()
        return PriceResult(
            price_jpy=price_value,
            stock_status=stock_status,
            scraped_at=scraped_at,
            title=title,
        )


class CollectorRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[], BaseCollector]] = {}
        self._fallback_factory: Callable[[], BaseCollector] = GenericHtmlCollector

    def register(self, shop: str, factory: Callable[[], BaseCollector]) -> None:
        self._registry[shop.lower()] = factory

    def set_fallback(self, factory: Callable[[], BaseCollector]) -> None:
        self._fallback_factory = factory

    def resolve(self, shop: Optional[str]) -> BaseCollector:
        if shop:
            key = shop.lower()
            if key in self._registry:
                return self._registry[key]()
        return self._fallback_factory()


registry = CollectorRegistry()


def collect_price(*, shop: Optional[str], url: str) -> PriceResult:
    collector = registry.resolve(shop)
    logger.info("collecting price", extra={"shop": shop, "collector": collector.name, "url": url})
    return collector.collect(url)


__all__ = ["PriceResult", "BaseCollector", "GenericHtmlCollector", "registry", "collect_price"]
