from __future__ import annotations

import json
from typing import Any


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def json_loads(s: str) -> Any:
    return json.loads(s)


def now_iso() -> str:
    from datetime import datetime
    return datetime.now().replace(microsecond=0).isoformat()
