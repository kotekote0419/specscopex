from __future__ import annotations

import requests

FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v1"


def fetch_usd_jpy_rates(start_date: str, end_date: str) -> dict[str, float]:
    """Fetch USD/JPY daily rates from Frankfurter.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        Mapping of date string to JPY rate. Empty dict when fetch fails.
    """

    url = f"{FRANKFURTER_BASE_URL}/{start_date}..{end_date}"
    params = {"base": "USD", "symbols": "JPY"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        return {
            date: float(value.get("JPY"))
            for date, value in rates.items()
            if isinstance(value, dict) and value.get("JPY") is not None
        }
    except requests.RequestException:
        return {}
    except ValueError:
        return {}
