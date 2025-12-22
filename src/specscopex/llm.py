from __future__ import annotations

from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .config import get_settings
from .schemas import UrlAuditV1, SkuMatchV1, AnomalyReviewV1, ExplanationV1

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMError("OPENAI_API_KEY is not set. Please set it in .env (or env vars).")
    return OpenAI(api_key=settings.openai_api_key)


def run_structured(text_format: Type[T], system: str, user: str, model: str | None = None) -> T:
    settings = get_settings()
    model_id = model or settings.openai_model
    try:
        resp = _client().responses.parse(
            model=model_id,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=text_format,
        )
        parsed: T = resp.output_parsed
        return parsed
    except Exception as e:
        raise LLMError(f"LLM structured call failed: {e}") from e


# ---- Task-specific helpers ----

def llm_url_audit(user_payload: str, model: str | None = None) -> UrlAuditV1:
    system = (
        "You are a strict auditor for a PC parts price tracker.\n"
        "You MUST only use the provided input fields; do not invent facts.\n"
        "If unsure, set needs_review=true and lower confidence.\n"
        "Output must follow the JSON schema."
    )
    user = (
        "Audit the following extracted info from a product page.\n"
        "Return whether it's a GPU product page and normalize fields.\n\n"
        f"{user_payload}"
    )
    return run_structured(UrlAuditV1, system, user, model=model)


def llm_sku_match(user_payload: str, model: str | None = None) -> SkuMatchV1:
    system = (
        "You help match shop listings to an internal SKU catalog for PC parts.\n"
        "Use only the provided candidates and extracted fields.\n"
        "If ambiguous, set match_type='unknown' and needs_review=true."
    )
    user = (
        "Decide whether this listing matches an existing SKU.\n\n"
        f"{user_payload}"
    )
    return run_structured(SkuMatchV1, system, user, model=model)


def llm_anomaly_review(user_payload: str, model: str | None = None) -> AnomalyReviewV1:
    system = (
        "You review suspicious price observations for PC parts.\n"
        "Use only provided numbers/text. Recommend keep/exclude/review.\n"
        "If unsure, recommended_action='review'."
    )
    user = (
        "Review this observation and decide if it should be kept.\n\n"
        f"{user_payload}"
    )
    return run_structured(AnomalyReviewV1, system, user, model=model)


def llm_explanation(user_payload: str, model: str | None = None) -> ExplanationV1:
    system = (
        "You write short, user-friendly explanations for buy timing decisions.\n"
        "You MUST not introduce external causes (FX/news) unless explicitly provided.\n"
        "Use the provided metrics only."
    )
    user = (
        "Generate a headline, short summary, 3 bullets, and a disclaimer.\n\n"
        f"{user_payload}"
    )
    return run_structured(ExplanationV1, system, user, model=model)


def llm_explain_signal(
    *, template_text: str, signals: dict, model: str | None = None, fx_summary: dict | None = None
) -> tuple[str, str]:
    settings = get_settings()
    model_id = model or settings.openai_model
    user = (
        "以下の買い時テンプレ根拠とシグナル要約をもとに、1〜2文の補足コメントを日本語で返してください。\n"
        "外部要因やセール、新製品などの可能性を推測する場合は控えめな表現にし、断定は避けてください。\n"
        "USD/JPYが変動している場合は、価格に影響する可能性として1文だけ触れて構いませんが、因果は断定せず外部要因の例として述べてください。\n"
        "渡された数値のみを根拠として使い、架空の数値は作らないでください。\n"
        "テンプレの内容と矛盾しないようにしてください。\n\n"
        f"テンプレ根拠: {template_text}\n"
        f"signals: {signals}\n"
        f"fx_summary: {fx_summary or 'unknown'}"
    )
    try:
        resp = _client().responses.create(
            model=model_id,
            input=[{"role": "user", "content": user}],
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise LLMError("LLM returned empty text")
        return text, model_id
    except Exception as e:
        raise LLMError(f"LLM explain call failed: {e}") from e
