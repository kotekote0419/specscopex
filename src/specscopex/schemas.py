from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field, conint, confloat


# --- A) URL監査 ---
class UrlAuditV1(BaseModel):
    schema_version: Literal["url_audit_v1"] = "url_audit_v1"
    is_gpu_page: bool
    normalized_model: Optional[str] = Field(default=None, max_length=80)
    variant: Optional[str] = Field(default=None, max_length=40)
    memory_gb: Optional[conint(ge=1, le=64)] = None
    condition: Literal["new", "used", "unknown"]
    bundle_suspected: bool
    price_type: Literal["tax_included", "tax_excluded", "unknown"]
    confidence: confloat(ge=0, le=1)
    issues: List[str] = Field(default_factory=list, max_length=10)
    needs_review: bool


# --- B) SKU同定 ---
class SkuMatchV1(BaseModel):
    schema_version: Literal["sku_match_v1"] = "sku_match_v1"
    match_type: Literal["new_sku", "alias_to_existing", "unknown"]
    matched_sku_id: Optional[str] = Field(default=None, max_length=64)
    normalized_model: Optional[str] = Field(default=None, max_length=80)
    variant: Optional[str] = Field(default=None, max_length=40)
    memory_gb: Optional[conint(ge=1, le=64)] = None
    confidence: confloat(ge=0, le=1)
    reasons: List[str] = Field(default_factory=list, max_length=10)
    needs_review: bool


# --- C) 異常精査 ---
class AnomalyReviewV1(BaseModel):
    schema_version: Literal["anomaly_review_v1"] = "anomaly_review_v1"
    is_outlier: bool
    outlier_reason: Literal[
        "none",
        "price_jump",
        "parse_suspected",
        "wrong_variant",
        "used_or_bundle",
        "shipping_points",
        "other",
    ]
    recommended_action: Literal["keep", "exclude", "review"]
    confidence: confloat(ge=0, le=1)
    notes: Optional[str] = Field(default=None, max_length=200)


# --- D) 説明生成 ---
class ExplanationV1(BaseModel):
    schema_version: Literal["explanation_v1"] = "explanation_v1"
    headline: str = Field(max_length=40)
    summary: str = Field(max_length=220)
    bullets: List[str] = Field(min_length=3, max_length=3)
    disclaimer: str = Field(max_length=160)
