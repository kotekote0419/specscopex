from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

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

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        """Parse int from typical price values (string/number), stripping commas."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            v = int(round(float(value)))
            return v
        if isinstance(value, str):
            s = value.strip()
            s = s.replace(",", "")
            # remove currency symbols/words
            s = s.replace("¥", "").replace("円", "").strip()
            # keep only digits and dot
            s = re.sub(r"[^\d.]", "", s)
            if not s:
                return None
            try:
                v = int(round(float(s)))
                return v
            except ValueError:
                return None
        return None

    @staticmethod
    def _is_plausible_price(v: int) -> bool:
        # GPU/PCパーツ用に「小さすぎる値」を弾きやすくする
        # （どうしても必要なら後で下限を下げられる）
        return 3_000 <= v <= 2_000_000

    def _extract_price_from_jsonld(self, soup: BeautifulSoup) -> Optional[int]:
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for sc in scripts:
            raw = sc.string or sc.get_text(strip=True)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            def iter_objs(obj: Any):
                if isinstance(obj, dict):
                    yield obj
                    for vv in obj.values():
                        yield from iter_objs(vv)
                elif isinstance(obj, list):
                    for it in obj:
                        yield from iter_objs(it)

            best: Optional[int] = None

            for obj in iter_objs(data):
                # Schema.org Product/Offer を想定
                if not isinstance(obj, dict):
                    continue

                # offers が dict or list のケースが多い
                offers = obj.get("offers")
                candidates: list[Any] = []
                if isinstance(offers, dict):
                    candidates.append(offers.get("price"))
                    candidates.append(offers.get("lowPrice"))
                    candidates.append(offers.get("highPrice"))
                elif isinstance(offers, list):
                    for off in offers:
                        if isinstance(off, dict):
                            candidates.append(off.get("price"))
                            candidates.append(off.get("lowPrice"))
                            candidates.append(off.get("highPrice"))

                for c in candidates:
                    v = self._to_int(c)
                    if v is None:
                        continue
                    if self._is_plausible_price(v):
                        # JSON-LDは信頼度高いので、見つかったら最初の妥当値を採用
                        return v
                    # もし下限で落ちた場合でも、bestとして保持だけ（最後の手段）
                    if 100 <= v <= 2_000_000:
                        best = v if best is None else max(best, v)

            if best is not None and self._is_plausible_price(best):
                return best

        return None

    def _extract_price_from_meta(self, soup: BeautifulSoup) -> Optional[int]:
        # itemprop price
        meta = soup.select_one('meta[itemprop="price"]')
        if meta and meta.get("content"):
            v = self._to_int(meta.get("content"))
            if v is not None and self._is_plausible_price(v):
                return v

        # OpenGraph / product meta
        for prop in ["product:price:amount", "og:price:amount", "og:price", "product:price"]:
            m = soup.find("meta", attrs={"property": prop})
            if m and m.get("content"):
                v = self._to_int(m.get("content"))
                if v is not None and self._is_plausible_price(v):
                    return v

        # Some sites use name=price
        for name in ["price", "twitter:data1"]:
            m = soup.find("meta", attrs={"name": name})
            if m and m.get("content"):
                v = self._to_int(m.get("content"))
                if v is not None and self._is_plausible_price(v):
                    return v

        return None

    def _extract_price_from_text(self, text: str) -> Optional[int]:
        # まず「¥/円が付いてる」価格候補を優先して拾う
        # 例: "¥123,456" / "123,456円"
        candidates: list[tuple[int, int]] = []  # (score, value)

        # 価格っぽいパターンを finditer して context でスコアリング
        # NOTE: 3〜8桁に限定（小さすぎる値は後で落とす）
        pattern = re.compile(r"(¥|\\u00a5)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{3,8})\s*(円)?")
        lower = text.lower()

        for m in pattern.finditer(text):
            raw_num = m.group(2)
            v = self._to_int(raw_num)
            if v is None:
                continue
            if not (100 <= v <= 2_000_000):
                continue

            # 周辺文脈
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            ctx = lower[start:end]

            score = 0
            token = (m.group(1) or "") + (m.group(3) or "")
            if "¥" in token or "円" in token:
                score += 8

            if any(k in ctx for k in ["販売価格", "価格", "税込", "円", "¥", "本体価格", "特価", "セール"]):
                score += 4

            # 「価格ではない」ノイズを強く減点
            if any(k in ctx for k in ["ポイント", "pt", "%", "％", "割引", "off", "クーポン", "送料", "送料無料"]):
                score -= 8
            if any(k in ctx for k in ["mhz", "ghz", "gb", "w", "mm", "cm", "インチ", "fps", "hz"]):
                score -= 3

            # GPU/パーツは高額なので、値が大きい方に少し寄せる
            if v >= 50_000:
                score += 2
            elif v < 5_000:
                score -= 6

            candidates.append((score, v))

        if not candidates:
            return None

        # まず「妥当レンジ（>=3000）」だけでベストを選ぶ。なければ全候補から。
        plausible = [(s, v) for (s, v) in candidates if self._is_plausible_price(v)]
        pool = plausible if plausible else candidates

        # score desc → value desc で決定
        pool.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = pool[0][1]
        return best if self._is_plausible_price(best) else None

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
        title = soup.title.get_text(" ", strip=True) if soup.title else None

        # 価格抽出：JSON-LD → meta → text
        price_value = self._extract_price_from_jsonld(soup)
        if price_value is None:
            price_value = self._extract_price_from_meta(soup)
        if price_value is None:
            page_text = soup.get_text(" ", strip=True)
            price_value = self._extract_price_from_text(page_text)
        else:
            page_text = soup.get_text(" ", strip=True)

        stock_status = self._extract_stock_status(page_text)
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
