from __future__ import annotations

from uuid import uuid4
from typing import Any

import pandas as pd
import requests
import streamlit as st

from specscopex.collectors.url_extract import fetch_and_extract
from specscopex.db import (
    ensure_schema,
    enqueue_review_item,
    find_alias_duplicate,
    find_product_by_key,
    get_review_item,
    insert_alias,
    insert_llm_audit,
    insert_product,
    upsert_product,
    upsert_product_url,
    list_aliases_for_sku,
    list_products,
    list_product_urls,
    list_product_urls_with_latest_price,
    set_product_url_active,
    delete_product_url,
    delete_product,
    list_review_items,
    save_review_draft_final,
    update_review_status,
    update_review_suggested,
)
from specscopex.llm import LLMError, llm_url_audit
from specscopex.utils import json_dumps, json_loads


st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")
ensure_schema()


# =========================================================
# Helpers
# =========================================================
def _status_emoji(status: str) -> str:
    return {"pending": "🟡", "approved": "✅", "rejected": "⛔"}.get(status, "•")


def _type_emoji(item_type: str) -> str:
    return {"sku_candidate": "🧩", "alias_candidate": "🔗"}.get(item_type, "📌")


def _safe_int(x) -> int | None:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def _coalesce(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None


def _payload_summary(item: dict[str, Any]) -> dict[str, Any]:
    payload = json_loads(item["payload_json"])
    it = item["item_type"]
    summary = {"title": "", "model": "", "url": "", "target_sku_id": ""}

    if it == "sku_candidate":
        extracted = payload.get("extracted", {})
        proposed = payload.get("proposed", {})
        summary["title"] = _coalesce(
            proposed.get("display_name"),
            extracted.get("page_h1"),
            extracted.get("page_title"),
            "sku_candidate",
        )
        summary["model"] = _coalesce(extracted.get("normalized_model"), "")
        summary["url"] = _coalesce(payload.get("source_url"), "")
        summary["target_sku_id"] = ""
    elif it == "alias_candidate":
        summary["title"] = "alias_candidate"
        summary["model"] = ""
        summary["url"] = _coalesce(payload.get("source_url"), payload.get("url"), "")
        summary["target_sku_id"] = _coalesce(payload.get("matched_sku_id"), "")
    else:
        summary["title"] = it
        summary["model"] = ""
        summary["url"] = ""
        summary["target_sku_id"] = ""

    return summary


def _product_label(p: dict[str, Any]) -> str:
    key = []
    if p.get("normalized_model"):
        key.append(p["normalized_model"])
    if p.get("variant"):
        key.append(p["variant"])
    if p.get("memory_gb") is not None:
        key.append(f"{p['memory_gb']}GB")
    suffix = " / ".join(key) if key else ""
    if suffix:
        return f"{p['sku_id']} | {p['display_name']}  ({suffix})"
    return f"{p['sku_id']} | {p['display_name']}"


def _sku_map(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p["sku_id"]: p for p in products if p.get("sku_id")}


def _norm_str(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    return s


def _count_duplicates(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        vv = _norm_str(v)
        if not vv:
            continue
        counts[vv] = counts.get(vv, 0) + 1
    return {k: c for k, c in counts.items() if c >= 2}


def _format_price_jpy(price: Any) -> str | None:
    try:
        if price is None:
            return None
        return f"¥{int(price):,}"
    except Exception:
        return None


def _format_latest_price(latest_price: Any, latest_scraped_at: Any) -> str:
    price_text = _format_price_jpy(latest_price)
    time_text = str(latest_scraped_at) if latest_scraped_at else None
    if not price_text and not time_text:
        return "最新取得: -（まだ取得なし）"
    return f"最新取得: {price_text or '-'} / {time_text or '-'}"


# =========================================================
# Session state
# =========================================================
st.session_state.setdefault("selected_review_id", None)
st.session_state.setdefault("inbox_selected_id", None)
st.session_state.setdefault("show_debug_json", False)
st.session_state.setdefault("confirm_action_token", None)

# ★安全な画面遷移用（widget keyではないのでいつでも変更OK）
st.session_state.setdefault("nav_target", None)

NAV_ADD = "➕ URL追加"
NAV_INBOX = "📥 Inbox"
NAV_REVIEW = "🧰 Review"
NAV_PRODUCTS = "📦 Products"
NAV_ITEMS = [NAV_ADD, NAV_INBOX, NAV_REVIEW, NAV_PRODUCTS]
st.session_state.setdefault("admin_nav", NAV_INBOX)

# ★次の実行の先頭で admin_nav に反映（radio作成前なのでOK）
if st.session_state.get("nav_target"):
    st.session_state["admin_nav"] = st.session_state["nav_target"]
    st.session_state["nav_target"] = None


# =========================================================
# Header + Navigation
# =========================================================
st.title("🛠️ Admin（かんたん管理画面）")
st.caption("導線：①URL追加 → ②Inboxで選択 → ③Reviewで処理（Approve/Reject/alias）")

st.radio("画面", NAV_ITEMS, horizontal=True, key="admin_nav")
st.divider()

nav = st.session_state["admin_nav"]


# =========================================================
# VIEW: Add URL
# =========================================================
if nav == NAV_ADD:
    st.subheader("➕ URL追加")

    # -----------------------------------------------------
    # (A) 既存SKUに URL を直接追加（即時反映）
    # -----------------------------------------------------
    st.markdown("### 1) 既存SKUにURLを追加（即時反映）")
    st.caption("URL解析/LLMは使いません。products（SKUマスタ）にあるSKUへ、収集対象URLを直接追加します。")

    prods_all = list_products(limit=2000)
    if not prods_all:
        st.info("products が空です。先に Products タブでSKUを作成してください。")
    else:
        sku_search = st.text_input(
            "SKU検索（display_name / model / sku_id）",
            value="",
            placeholder="例: 4070 / ASUS / NVIDIA_RTX...",
            key="add_url_sku_search",
        )

        def _match(p: dict) -> bool:
            if not sku_search.strip():
                return True
            q = sku_search.strip().lower()
            return (
                q in (p.get("sku_id") or "").lower()
                or q in (p.get("display_name") or "").lower()
                or q in (p.get("normalized_model") or "").lower()
            )

        prods = [p for p in prods_all if _match(p)]
        prods = prods[:200]  # UI重くしないため上限

        options = {
            f"{p.get('display_name','(no name)')}  |  {p['sku_id']}": p["sku_id"]
            for p in prods
        }

        if not options:
            st.warning("検索条件に一致するSKUがありません。")
        else:
            c1, c2 = st.columns([2, 1], gap="large")
            with c1:
                selected_label = st.selectbox("SKUを選択", list(options.keys()), key="add_url_existing_sku")
                selected_sku = options[selected_label]

                shop_direct = st.text_input(
                    "shop（ラベル）",
                    value="dospara",
                    placeholder="dospara / tsukumo / ark など（いまはラベル扱い）",
                    key="add_url_shop_direct",
                )
                url_direct = st.text_input(
                    "製品URL",
                    value="",
                    placeholder="例: https://.../item/xxxx",
                    key="add_url_input_direct",
                )
                is_active_direct = st.checkbox("収集対象にする（is_active=true）", value=True, key="add_url_active_direct")

            with c2:
                st.markdown("#### メモ")
                st.markdown("- shopは現状ラベル扱い（Generic collectorで収集）")
                st.markdown("- URLは同一SKUでも複数ショップ追加OK")
                st.markdown("- 重複はUNIQUEで抑制されます")

            save_direct = st.button(
                "URLを登録（即時反映）",
                type="primary",
                disabled=(not shop_direct.strip() or not url_direct.strip()),
                key="add_url_save_direct",
            )

            if save_direct:
                try:
                    pid = upsert_product_url(
                        sku_id=selected_sku,
                        shop=shop_direct.strip(),
                        url=url_direct.strip(),
                        title=None,
                        is_active=bool(is_active_direct),
                    )
                    st.success(f"登録しました（product_url_id={pid}）")
                    st.rerun()
                except Exception as e:
                    st.error(f"登録に失敗しました: {e}")

    st.divider()

    # -----------------------------------------------------
    # (B) URL → LLM監査 → Inbox（従来フロー）
    # -----------------------------------------------------
    st.markdown("### 2) URLからSKU候補を作成（分析→Inbox）")
    st.caption("URLの内容を抽出し、LLMで監査して Inbox に積みます（Reviewで承認/却下）。")

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        url = st.text_input("製品URL", value="", placeholder="例: https://.../item/xxxx", key="add_url_input_llm")
        shop = st.text_input("ショップ名（任意）", value="", placeholder="dospara / tsukumo / ark など", key="add_shop_input_llm")
    with col2:
        st.markdown("#### コツ")
        st.markdown("- まずは専門店だけでOK\n- セット品/中古っぽい場合はReviewでReject")

    add_btn = st.button(
        "追加する（分析→Inboxへ）",
        type="secondary",
        disabled=(not url.strip()),
        key="add_submit_btn_llm",
    )

    if add_btn:
        try:
            with st.spinner("ページ取得 & 監査中..."):
                ext = fetch_and_extract(url.strip())

                payload = {
                    "url": ext.url,
                    "shop": shop.strip() or None,
                    "page": {
                        "title": ext.title,
                        "h1": ext.h1,
                        "text_snippet": ext.text_snippet,
                    },
                }
                payload_str = json_dumps(payload)

                audit = llm_url_audit(payload_str)

                insert_llm_audit(
                    task_type="url_audit",
                    model_id=None,
                    prompt_version="p1",
                    schema_version=audit.schema_version,
                    input_digest=payload_str,
                    output_json=audit.raw_json,
                    confidence=float(audit.confidence),
                    needs_review=bool(audit.needs_review),
                )

                extracted = audit.extracted or {}
                suggested = audit.suggested or {}

                sku_payload = {
                    "source_url": ext.url,
                    "source_shop": shop.strip() or None,
                    "page_title": ext.title,
                    "page_h1": ext.h1,
                    "extracted": extracted,
                    "proposed": {
                        "sku_id": suggested.get("sku_id") or "",
                        "normalized_model": suggested.get("normalized_model") or "",
                        "variant": suggested.get("variant") or None,
                        "memory_gb": suggested.get("memory_gb") or None,
                        "display_name": suggested.get("normalized_model") or (ext.h1 or ext.title or "Unknown Part"),
                        "perf_score": None,
                    },
                }

                item_id = enqueue_review_item(
                    item_type="sku_candidate",
                    payload_obj=sku_payload,
                    suggested_obj=suggested,
                    confidence=float(audit.confidence),
                    needs_review=bool(audit.needs_review),
                    model_id=None,
                    prompt_version="p1",
                    schema_version=audit.schema_version,
                )

            st.success(f"Inboxに追加しました：#{item_id}（sku_candidate）")
            st.session_state["selected_review_id"] = item_id
            st.session_state["inbox_selected_id"] = item_id

            if st.button("このまま Review を開く", type="primary", key=f"add_open_review_{item_id}"):
                st.session_state["nav_target"] = NAV_REVIEW
                st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"URL取得に失敗しました: {e}")
        except LLMError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    # =========================================================
    # (C) URL管理（既存URLの一覧・有効/無効・削除）
    # =========================================================
    st.divider()
    st.subheader("🔧 URL管理（既存URLの有効/無効・削除）")
    st.caption("※無効化すると収集ジョブの対象外になります。削除するとそのURLの価格履歴も消えます。")

    products_all = list_products(limit=2000)
    if not products_all:
        st.info("products が空です。先に Products タブでSKUを作成してください。")
    else:
        # _product_label() はこのファイル上部で定義されている前提（既に使ってるはず）
        labels = [_product_label(p) for p in products_all]
        sku_by_label = {_product_label(p): p["sku_id"] for p in products_all}

        selected_label_mgmt = st.selectbox("対象SKU", labels, index=0, key="url_mgmt_sku")
        selected_sku_id_mgmt = sku_by_label.get(selected_label_mgmt)

        include_inactive = st.toggle("無効URLも表示", value=True, key="url_mgmt_include_inactive")

        try:
            urls = list_product_urls_with_latest_price(
                sku_id=selected_sku_id_mgmt,
                include_inactive=include_inactive,
                limit=500,
            )
        except Exception:
            urls = list_product_urls(
                sku_id=selected_sku_id_mgmt,
                include_inactive=include_inactive,
                limit=500,
            )

        if not urls:
            st.info("このSKUにはURLがまだ登録されていません。上のフォームから追加してください。")
        else:
            for row in urls:
                pid = int(row["id"])
                shop_v = (row.get("shop") or "").strip()
                url_v = (row.get("url") or "").strip()
                title_v = (row.get("title") or "").strip()
                active_v = bool(row.get("is_active"))

                badge = "🟢 Active" if active_v else "⚪ Inactive"
                st.markdown(f"**{badge}**  `#{pid}`  **{shop_v or '(shop未設定)'}**")
                st.write(url_v)
                if title_v:
                    st.caption(title_v)

                latest_price = row.get("latest_price_jpy")
                latest_scraped_at = row.get("latest_scraped_at")
                st.caption(_format_latest_price(latest_price, latest_scraped_at))

                c1, c2, c3 = st.columns([1.2, 1.0, 6.0], gap="small")

                with c1:
                    if st.button("無効化" if active_v else "有効化", key=f"url_toggle_{pid}"):
                        set_product_url_active(product_url_id=pid, is_active=(not active_v))
                        st.success("更新しました。")
                        st.rerun()

                with c2:
                    if st.button("削除", key=f"url_delete_{pid}"):
                        st.session_state["confirm_action_token"] = f"delete_purl_{pid}"
                        st.rerun()

                with c3:
                    if st.session_state.get("confirm_action_token") == f"delete_purl_{pid}":
                        st.warning("このURLを削除します。配下の price_history も消えます。本当に削除しますか？")
                        cc1, cc2 = st.columns([1, 1], gap="small")
                        with cc1:
                            if st.button("削除確定", type="primary", key=f"url_delete_confirm_{pid}"):
                                delete_product_url(product_url_id=pid)
                                st.session_state["confirm_action_token"] = None
                                st.success("削除しました。")
                                st.rerun()
                        with cc2:
                            if st.button("キャンセル", key=f"url_delete_cancel_{pid}"):
                                st.session_state["confirm_action_token"] = None
                                st.rerun()

                st.divider()

# =========================================================
# VIEW: Inbox
# =========================================================
elif nav == NAV_INBOX:
    st.subheader("📥 Inbox（やることリスト）")
    st.caption("※単一選択のみ（Selectチェックは廃止）")

    # A) alias紐付け先を表示するため、productsを先読み（存在しないSKUでも壊れない）
    products_all = list_products(limit=2000)
    sku_map = _sku_map(products_all)

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1], gap="large")
    with f1:
        status = st.selectbox("status", ["pending", "approved", "rejected", "(all)"], index=0, key="inbox_status")
    with f2:
        item_type = st.text_input("type（空=全件）", value="", placeholder="sku_candidate / alias_candidate", key="inbox_type")
    with f3:
        limit = st.slider("表示件数", 50, 500, 200, step=50, key="inbox_limit")
    with f4:
        _ = st.button("更新", key="inbox_refresh_btn")

    status_filter = None if status == "(all)" else status
    item_type_filter = item_type.strip() or None

    items = list_review_items(status=status_filter, item_type=item_type_filter, limit=limit)

    if not items:
        st.info("該当データがありません。URL追加から作ってください。")
    else:
        rows = []
        options = []
        for it in items:
            s = _payload_summary(it)
            rid = int(it["id"])

            target_sku = s.get("target_sku_id") or ""
            target_name = ""
            if target_sku and target_sku in sku_map:
                target_name = sku_map[target_sku].get("display_name") or ""

            rows.append(
                {
                    "ID": rid,
                    "Status": f"{_status_emoji(it['status'])} {it['status']}",
                    "Type": f"{_type_emoji(it['item_type'])} {it['item_type']}",
                    "Conf": float(it["confidence"]) if it.get("confidence") is not None else None,
                    "Review?": bool(it["needs_review"]) if it.get("needs_review") is not None else False,
                    "Target SKU": target_sku,
                    "Target Name": target_name,
                    "Title": s["title"],
                    "Model": s["model"],
                    "URL": s["url"],
                    # ソート用（見せない）
                    "_is_alias": 1 if it["item_type"] == "alias_candidate" else 0,
                }
            )

            # Selectbox label（aliasならTargetを目立たせる）
            if it["item_type"] == "alias_candidate":
                label = f"#{rid} | alias → {target_name or target_sku or '???'}"
            else:
                label = f"#{rid} | {s['title']} | {s['model']}"
            options.append((rid, label))

        df = pd.DataFrame(rows)

        # ★追加改善：alias_candidate を Target Name でまとまるようにソート
        # （aliasを先に、Target Name → Target SKU → ID）
        # _is_alias: alias=1, others=0 なので、降順でaliasが上に来る
        if "Target Name" in df.columns:
            df["Target Name"] = df["Target Name"].fillna("")
        if "Target SKU" in df.columns:
            df["Target SKU"] = df["Target SKU"].fillna("")

        df = df.sort_values(
            by=["_is_alias", "Target Name", "Target SKU", "ID"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )

        # 表示用（内部列は落とす）
        df_view = df.drop(columns=["_is_alias"], errors="ignore")

        st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Conf": st.column_config.NumberColumn("Conf", format="%.2f"),
                "Review?": st.column_config.CheckboxColumn("Review?"),
            },
        )

        ids = [x[0] for x in options]
        labels = {x[0]: x[1] for x in options}

        default_id = st.session_state.get("inbox_selected_id") or st.session_state.get("selected_review_id") or ids[0]
        if default_id not in ids:
            default_id = ids[0]

        chosen_id = st.selectbox(
            "開くID（単一選択）",
            ids,
            index=ids.index(default_id),
            format_func=lambda x: labels.get(x, str(x)),
            key="inbox_single_selectbox",
        )
        st.session_state["inbox_selected_id"] = int(chosen_id)

        if st.button("🧰 Reviewで開く", type="primary", key="inbox_open_btn"):
            st.session_state["selected_review_id"] = int(chosen_id)
            st.session_state["nav_target"] = NAV_REVIEW
            st.rerun()


# =========================================================
# VIEW: Review
# =========================================================
elif nav == NAV_REVIEW:
    st.subheader("🧰 Review（ここだけ見ればOK）")

    selected_id = st.session_state.get("selected_review_id")
    if not selected_id:
        st.info("InboxでIDを選択して「Reviewで開く」を押してください。")
    else:
        item = get_review_item(int(selected_id))
        if not item:
            st.error("選択アイテムが見つかりません。")
        else:
            header_cols = st.columns([1.2, 1, 1, 1, 1.2], gap="large")
            header_cols[0].markdown(f"**ID**: `{item['id']}`")
            header_cols[1].markdown(f"**Type**: `{item['item_type']}`")
            header_cols[2].markdown(f"**Status**: `{item['status']}`")
            header_cols[3].markdown(f"**needs_review**: `{bool(item.get('needs_review')) if item.get('needs_review') is not None else '-'}'")
            header_cols[4].markdown(f"**confidence**: `{(float(item['confidence']) if item.get('confidence') is not None else '-')}`")

            payload_obj = json_loads(item["payload_json"])
            draft_obj = json_loads(item["final_json"]) if item.get("final_json") else None

            st.divider()

            resolver = st.text_input("処理者（resolver）", value="admin", key=f"resolver_{item['id']}")

            actA, actB, actC, actD, actE = st.columns([1, 1, 1, 2, 1.2], gap="large")
            with actA:
                btn_reject = st.button("⛔ Reject", key=f"review_reject_{item['id']}")
            with actB:
                btn_reopen = st.button("↩️ Reopen", key=f"review_reopen_{item['id']}")
            with actC:
                btn_rerun = st.button("🔁 Re-run LLM", key=f"review_rerun_{item['id']}")
            with actD:
                st.session_state["show_debug_json"] = st.toggle(
                    "デバッグJSONを表示",
                    value=st.session_state.get("show_debug_json", False),
                    key=f"toggle_debug_{item['id']}",
                )
            with actE:
                if st.button("📥 Inboxへ戻る", key=f"review_back_inbox_{item['id']}"):
                    st.session_state["nav_target"] = NAV_INBOX
                    st.rerun()

            if btn_reject:
                update_review_status(
                    item_id=item["id"],
                    new_status="rejected",
                    resolver=resolver,
                    reason_code="manual_reject",
                    note="rejected in easy admin",
                )
                st.success("rejected")
                st.session_state["nav_target"] = NAV_INBOX
                st.rerun()

            if btn_reopen:
                update_review_status(
                    item_id=item["id"],
                    new_status="pending",
                    resolver=resolver,
                    note="reopened in easy admin",
                )
                st.success("reopened")
                st.rerun()

            if btn_rerun:
                try:
                    payload_str = json_dumps(payload_obj)
                    audit = llm_url_audit(payload_str)
                    suggested = audit.model_dump()

                    insert_llm_audit(
                        task_type="url_audit",
                        model_id=None,
                        prompt_version="p1",
                        schema_version=audit.schema_version,
                        input_digest=payload_str[:5000],
                        output_json=json_dumps(suggested),
                        confidence=float(audit.confidence),
                        needs_review=bool(audit.needs_review),
                    )

                    update_review_suggested(
                        item_id=item["id"],
                        suggested_obj=suggested,
                        confidence=float(audit.confidence),
                        needs_review=bool(audit.needs_review),
                        model_id=None,
                        prompt_version="p1",
                        schema_version=audit.schema_version,
                    )
                    st.success("Re-run LLM done")
                    st.rerun()
                except LLMError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Re-run failed: {e}")

            st.divider()

            # -------------------------
            # sku_candidate
            # -------------------------
            if item["item_type"] == "sku_candidate":
                base = draft_obj or payload_obj
                extracted = base.get("extracted", payload_obj.get("extracted", {}))
                proposed = base.get("proposed", payload_obj.get("proposed", {}))

                st.markdown("### sku_candidate（新規SKU候補）")

                info_cols = st.columns([2, 1], gap="large")
                with info_cols[0]:
                    st.write("ページ情報（参考）")
                    st.code(
                        {
                            "page_h1": extracted.get("page_h1"),
                            "page_title": extracted.get("page_title"),
                            "source_url": base.get("source_url"),
                            "shop": base.get("shop"),
                        }
                    )
                with info_cols[1]:
                    flags = []
                    if extracted.get("is_gpu_page") is False:
                        flags.append("⚠️ GPUページではない可能性")
                    if extracted.get("bundle_suspected"):
                        flags.append("⚠️ セット品/バンドル疑い")
                    if extracted.get("condition") == "used":
                        flags.append("⚠️ 中古の可能性")
                    if flags:
                        for f in flags:
                            st.warning(f)
                    else:
                        st.success("大きな警告はありません")

                form_key = f"sku_form_{item['id']}"
                with st.form(form_key, clear_on_submit=False):
                    c1, c2 = st.columns([1.2, 1], gap="large")

                    with c1:
                        default_source_url = base.get("source_url") or ""
                        default_shop = base.get("shop")

                        st.text_input("source_url", value=default_source_url, disabled=True, key=f"{form_key}_srcurl")
                        shop_in = st.text_input("shop（任意）", value=default_shop or "", key=f"{form_key}_shop")

                        display_name = st.text_input(
                            "display_name（表示名）",
                            value=_coalesce(proposed.get("display_name"), extracted.get("normalized_model"), extracted.get("page_h1"), "Unknown Part"),
                            key=f"{form_key}_display",
                        )
                        normalized_model = st.text_input(
                            "normalized_model（必須推奨）",
                            value=extracted.get("normalized_model") or "",
                            key=f"{form_key}_model",
                        )
                        variant = st.text_input("variant（任意）", value=extracted.get("variant") or "", key=f"{form_key}_variant")

                    with c2:
                        memory_gb = st.number_input(
                            "memory_gb（任意 / 不明なら0）",
                            min_value=0,
                            max_value=64,
                            value=_safe_int(extracted.get("memory_gb")) or 0,
                            step=1,
                            key=f"{form_key}_mem",
                        )
                        perf_score = st.number_input(
                            "perf_score（任意 / 不明なら0）",
                            min_value=0.0,
                            value=float(proposed.get("perf_score") or 0.0),
                            step=100.0,
                            key=f"{form_key}_perf",
                        )

                        is_gpu_page = st.checkbox(
                            "is_gpu_page",
                            value=bool(extracted.get("is_gpu_page")) if extracted.get("is_gpu_page") is not None else False,
                            key=f"{form_key}_isgpu",
                        )
                        condition = st.selectbox(
                            "condition",
                            ["new", "used", "unknown"],
                            index=["new", "used", "unknown"].index(extracted.get("condition") or "unknown"),
                            key=f"{form_key}_cond",
                        )
                        bundle_suspected = st.checkbox(
                            "bundle_suspected",
                            value=bool(extracted.get("bundle_suspected")) if extracted.get("bundle_suspected") is not None else False,
                            key=f"{form_key}_bundle",
                        )
                        price_type = st.selectbox(
                            "price_type",
                            ["tax_included", "tax_excluded", "unknown"],
                            index=["tax_included", "tax_excluded", "unknown"].index(extracted.get("price_type") or "unknown"),
                            key=f"{form_key}_pricetype",
                        )

                    new_payload = {
                        "source_url": default_source_url,
                        "shop": shop_in.strip() or None,
                        "extracted": {
                            "page_title": extracted.get("page_title"),
                            "page_h1": extracted.get("page_h1"),
                            "normalized_model": normalized_model.strip() or None,
                            "variant": variant.strip() or None,
                            "memory_gb": None if memory_gb == 0 else int(memory_gb),
                            "condition": condition,
                            "bundle_suspected": bool(bundle_suspected),
                            "price_type": price_type,
                            "is_gpu_page": bool(is_gpu_page),
                        },
                        "proposed": {
                            "display_name": display_name.strip(),
                            "perf_score": None if perf_score == 0.0 else float(perf_score),
                        },
                    }

                    dup = find_product_by_key(
                        normalized_model=new_payload["extracted"].get("normalized_model"),
                        variant=new_payload["extracted"].get("variant"),
                        memory_gb=new_payload["extracted"].get("memory_gb"),
                    )

                    if dup:
                        st.info(f"重複候補：{dup['sku_id']} / {dup['display_name']}")
                        action_choice = st.radio(
                            "処理方法（推奨：alias）",
                            ["既存SKUにURLを紐付け（alias：推奨）", "新規SKUとして登録"],
                            index=0,
                            key=f"{form_key}_choice",
                        )
                    else:
                        action_choice = "新規SKUとして登録"

                    st.markdown("---")
                    save_draft = st.form_submit_button("💾 下書き保存", use_container_width=True)
                    approve = st.form_submit_button("✅ Approve", use_container_width=True)

                if save_draft:
                    save_review_draft_final(item_id=item["id"], final_obj=new_payload)
                    st.success("下書きを保存しました（final_json）")
                    st.rerun()

                if approve:
                    if not new_payload["extracted"].get("is_gpu_page", False):
                        st.error("is_gpu_page=false のためApproveできません（Reject推奨）。")
                        st.stop()

                    dup2 = find_product_by_key(
                        normalized_model=new_payload["extracted"].get("normalized_model"),
                        variant=new_payload["extracted"].get("variant"),
                        memory_gb=new_payload["extracted"].get("memory_gb"),
                    )

                    if dup2 and action_choice.startswith("既存SKU"):
                        alias_payload = {
                            "matched_sku_id": dup2["sku_id"],
                            "source_url": new_payload.get("source_url"),
                            "shop": new_payload.get("shop"),
                            "alias_text": new_payload.get("proposed", {}).get("display_name"),
                            "from_review_id": item["id"],
                            "hint": {
                                "normalized_model": new_payload["extracted"].get("normalized_model"),
                                "variant": new_payload["extracted"].get("variant"),
                                "memory_gb": new_payload["extracted"].get("memory_gb"),
                            },
                        }

                        alias_item_id = enqueue_review_item(
                            item_type="alias_candidate",
                            payload_obj=alias_payload,
                            suggested_obj=None,
                            confidence=item.get("confidence"),
                            needs_review=True,
                            model_id=item.get("model_id"),
                            prompt_version=item.get("prompt_version"),
                            schema_version="alias_candidate_v1",
                        )

                        update_review_status(
                            item_id=item["id"],
                            new_status="rejected",
                            resolver=resolver,
                            reason_code="duplicate_converted",
                            note=f"converted to alias_candidate #{alias_item_id}",
                        )

                        st.success(f"alias_candidate を作成：#{alias_item_id}（元はrejected）")
                        st.session_state["selected_review_id"] = alias_item_id
                        st.session_state["inbox_selected_id"] = alias_item_id
                        st.rerun()

                    if dup2 and action_choice.startswith("新規SKU"):
                        token = f"confirm_newsku_{item['id']}"
                        if st.session_state.get("confirm_action_token") != token:
                            st.session_state["confirm_action_token"] = token
                            st.warning("重複候補あり。もう一度Approveで『新規SKU作成』します。")
                            st.stop()

                    sku_id = f"sku_{uuid4().hex}"
                    display_name2 = new_payload["proposed"].get("display_name") or new_payload["extracted"].get("normalized_model") or "Unknown Part"

                    insert_product(
                        sku_id=sku_id,
                        display_name=display_name2,
                        normalized_model=new_payload["extracted"].get("normalized_model"),
                        variant=new_payload["extracted"].get("variant"),
                        memory_gb=new_payload["extracted"].get("memory_gb"),
                        perf_score=new_payload["proposed"].get("perf_score"),
                    )
                    insert_alias(
                        sku_id=sku_id,
                        shop=new_payload.get("shop"),
                        alias_text=display_name2,
                        url=new_payload.get("source_url"),
                    )

                    final_saved = {**new_payload, "approved_product": {"sku_id": sku_id, "display_name": display_name2}}

                    update_review_status(
                        item_id=item["id"],
                        new_status="approved",
                        resolver=resolver,
                        final_obj=final_saved,
                        note="approved -> products inserted (easy admin)",
                    )
                    st.success("approved（productsに登録しました）")
                    st.session_state["confirm_action_token"] = None
                    st.session_state["nav_target"] = NAV_INBOX
                    st.rerun()

            # -------------------------
            # alias_candidate
            # -------------------------
            elif item["item_type"] == "alias_candidate":
                base = draft_obj or payload_obj
                products = list_products(limit=2000)
                if not products:
                    st.error("products が空です。先に sku_candidate をApproveしてください。")
                    st.stop()

                target_sku_id = base.get("matched_sku_id")
                sku_map2 = _sku_map(products)
                target_name = (sku_map2.get(target_sku_id, {}) or {}).get("display_name") if target_sku_id else ""
                if target_sku_id:
                    st.info(f"紐付け先（現在）：{target_name or ''}  /  {target_sku_id}")

                st.markdown("### alias_candidate（既存SKUにURL/別名を追加）")

                labels = [_product_label(p) for p in products]
                sku_ids = [p["sku_id"] for p in products]
                default_sku = base.get("matched_sku_id")
                default_index = sku_ids.index(default_sku) if default_sku in sku_ids else 0

                form_key = f"alias_form_{item['id']}"
                with st.form(form_key, clear_on_submit=False):
                    sku_sel = st.selectbox(
                        "紐付け先SKU",
                        list(range(len(labels))),
                        index=default_index,
                        format_func=lambda i: labels[i],
                        key=f"{form_key}_skusel",
                    )
                    chosen_sku_id = sku_ids[sku_sel]

                    colL, colR = st.columns([1.2, 1], gap="large")
                    with colL:
                        shop_in = st.text_input("shop（任意）", value=(base.get("shop") or ""), key=f"{form_key}_shop")
                        url_in = st.text_input("url（任意）", value=_coalesce(base.get("source_url"), base.get("url"), "") or "", key=f"{form_key}_url")
                        alias_text_in = st.text_input("alias_text（任意）", value=(base.get("alias_text") or ""), key=f"{form_key}_aliastext")
                    with colR:
                        st.markdown("#### 既存alias（参考）")
                        aliases = list_aliases_for_sku(sku_id=chosen_sku_id, limit=200)
                        if aliases:
                            st.dataframe(
                                [{"shop": a.get("shop"), "alias_text": a.get("alias_text"), "url": a.get("url")} for a in aliases],
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.caption("alias はありません。")

                    new_payload = {
                        "matched_sku_id": chosen_sku_id,
                        "shop": shop_in.strip() or None,
                        "url": url_in.strip() or None,
                        "alias_text": alias_text_in.strip() or None,
                        "from_review_id": base.get("from_review_id"),
                        "hint": base.get("hint"),
                    }

                    dup_alias = find_alias_duplicate(
                        sku_id=chosen_sku_id,
                        url=new_payload.get("url"),
                        shop=new_payload.get("shop"),
                        alias_text=new_payload.get("alias_text"),
                    )
                    if dup_alias:
                        st.warning("同一aliasの可能性があります（重複登録注意）。")

                    st.markdown("---")
                    save_draft = st.form_submit_button("💾 下書き保存", use_container_width=True)
                    approve = st.form_submit_button("✅ Approve（alias追加）", use_container_width=True)

                if save_draft:
                    save_review_draft_final(item_id=item["id"], final_obj=new_payload)
                    st.success("下書きを保存しました（final_json）")
                    st.rerun()

                if approve:
                    dup_alias2 = find_alias_duplicate(
                        sku_id=new_payload["matched_sku_id"],
                        url=new_payload.get("url"),
                        shop=new_payload.get("shop"),
                        alias_text=new_payload.get("alias_text"),
                    )
                    if dup_alias2:
                        st.error("aliasが重複しそうです。内容を変更するかRejectしてください。")
                        st.stop()

                    insert_alias(
                        sku_id=new_payload["matched_sku_id"],
                        shop=new_payload.get("shop"),
                        alias_text=new_payload.get("alias_text"),
                        url=new_payload.get("url"),
                    )

                    final_saved = {
                        **new_payload,
                        "approved_alias": {
                            "sku_id": new_payload["matched_sku_id"],
                            "shop": new_payload.get("shop"),
                            "alias_text": new_payload.get("alias_text"),
                            "url": new_payload.get("url"),
                        },
                    }

                    update_review_status(
                        item_id=item["id"],
                        new_status="approved",
                        resolver=resolver,
                        final_obj=final_saved,
                        note="approved -> alias inserted (easy admin)",
                    )
                    st.success("approved（aliasを追加しました）")
                    st.session_state["nav_target"] = NAV_INBOX
                    st.rerun()

            if st.session_state.get("show_debug_json"):
                st.divider()
                st.subheader("🧪 デバッグJSON（通常はOFFでOK）")
                st.code(item["payload_json"], language="json")
                st.code(item.get("suggested_json") or "null", language="json")
                st.code(item.get("final_json") or "null", language="json")


# =========================================================
# VIEW: Products  (B: SKU→aliasがその場で見える + 重複警告)
# =========================================================
elif nav == NAV_PRODUCTS:
    st.subheader("📦 Products（SKU → alias一覧）")
    st.caption("各SKUの下で alias をすぐ確認できます（expander）。重複（URL/alias_text）も警告します。")

    # =========================================================
    # ★追加：URL不要のSKU登録（手動）フォーム
    # =========================================================
    st.markdown("### ➕ SKUを追加 / 更新（URL不要）")
    st.caption("seed(json)と同じ項目でSKUマスタ(products)を作れます。URLはあとで『➕ URL追加』から入れればOKです。")

    with st.form("manual_sku_upsert_form", clear_on_submit=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            sku_id_in = st.text_input(
                "sku_id *",
                placeholder="NVIDIA_RTX4070SUPER_ASUS_DUAL_12G など",
            )
            display_name_in = st.text_input(
                "display_name *",
                placeholder="ASUS Dual GeForce RTX 4070 SUPER 12GB など",
            )
            normalized_model_in = st.text_input(
                "normalized_model（推奨）",
                placeholder="RTX 4070 SUPER など",
            )

        with c2:
            variant_in = st.text_input("variant（任意）", placeholder="ASUS DUAL など")
            memory_gb_in = st.number_input("memory_gb（不明なら0）", min_value=0, max_value=64, value=0, step=1)
            perf_score_in = st.number_input("perf_score（不明なら0）", min_value=0.0, value=0.0, step=100.0)

        submitted = st.form_submit_button("保存（SKU追加/更新）", use_container_width=True)

        if submitted:
            try:
                upsert_product(
                    sku_id=(sku_id_in or "").strip(),
                    display_name=(display_name_in or "").strip(),
                    normalized_model=(normalized_model_in or "").strip() or None,
                    variant=(variant_in or "").strip() or None,
                    memory_gb=None if int(memory_gb_in) == 0 else int(memory_gb_in),
                    perf_score=None if float(perf_score_in) == 0.0 else float(perf_score_in),
                )
                st.success("SKUを保存しました。下の一覧で確認できます。")
                st.rerun()
            except Exception as e:
                st.error(f"保存に失敗しました: {e}")

    # URL追加へ誘導（任意）
    col_go, _ = st.columns([1, 3])
    with col_go:
        if st.button("➕ URL追加へ移動", use_container_width=True, key="goto_add_url_from_products"):
            st.session_state["nav_target"] = NAV_ADD
            st.rerun()

    st.divider()


    topL, topR = st.columns([1, 2], gap="large")
    with topL:
        limit = st.slider("表示件数", 50, 2000, 300, step=50, key="products_limit")
    with topR:
        q = st.text_input("検索（display_name / model / sku_id）", value="", placeholder="例: RTX 4070 / sku_... / 16GB", key="products_search")

    prods = list_products(limit=limit)
    if not prods:
        st.info("products がありません。sku_candidate を Approve してください。")
    else:
        # 検索フィルタ
        if q.strip():
            qq = q.strip().lower()
            filtered = []
            for p in prods:
                blob = " ".join(
                    [
                        str(p.get("sku_id", "")),
                        str(p.get("display_name", "")),
                        str(p.get("normalized_model", "")),
                        str(p.get("variant", "")),
                        str(p.get("memory_gb", "")),
                    ]
                ).lower()
                if qq in blob:
                    filtered.append(p)
            prods = filtered

        # aliases を SKUごとに1回だけ取得（重複チェックもここから）
        alias_cache: dict[str, list[dict[str, Any]]] = {}
        for p in prods:
            sku_id = p["sku_id"]
            alias_cache[sku_id] = list_aliases_for_sku(sku_id=sku_id, limit=2000)

        # ★追加改善：SKUをまたいだURL重複の検出（同じURLが複数SKUに紐付く）
        url_to_skus: dict[str, set[str]] = {}
        for sku_id, aliases in alias_cache.items():
            for a in aliases:
                url = _norm_str(a.get("url"))
                if not url:
                    continue
                url_to_skus.setdefault(url, set()).add(sku_id)

        cross_url_dups = [(url, sorted(list(skus))) for url, skus in url_to_skus.items() if len(skus) >= 2]
        if cross_url_dups:
            st.warning(f"⚠️ URLが複数SKUに紐付いています（{len(cross_url_dups)}件）。誤紐付けの可能性あり。")
            with st.expander("重複URL一覧（SKUまたぎ）", expanded=False):
                st.dataframe(
                    [{"url": url, "sku_ids": ", ".join(skus)} for url, skus in cross_url_dups],
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### SKU一覧（alias数つき）")
        preview_rows = []
        for p in prods:
            sku_id = p["sku_id"]
            aliases_preview = alias_cache.get(sku_id, [])
            preview_rows.append(
                {
                    "sku_id": sku_id,
                    "display_name": p.get("display_name"),
                    "normalized_model": p.get("normalized_model"),
                    "variant": p.get("variant"),
                    "memory_gb": p.get("memory_gb"),
                    "perf_score": p.get("perf_score"),
                    "alias_count": len(aliases_preview),
                }
            )

        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### SKUごとの alias 詳細")

        for p in prods:
            sku_id = p["sku_id"]
            display_name = p.get("display_name") or sku_id

            aliases = alias_cache.get(sku_id, [])
            alias_count = len(aliases)

            header = f"{display_name}  —  {sku_id}   (aliases: {alias_count})"
            with st.expander(header, expanded=False):
                delete_disabled = alias_count > 0
                delete_reason = "aliases または URLs が残っている場合は削除できません。" if delete_disabled else ""
                delete_cols = st.columns([1.2, 3], gap="small")
                with delete_cols[0]:
                    if st.button("🗑️ SKU削除", disabled=delete_disabled, key=f"sku_delete_{sku_id}"):
                        st.session_state["confirm_action_token"] = f"delete_sku_{sku_id}"
                        st.rerun()
                with delete_cols[1]:
                    if delete_reason:
                        st.caption(delete_reason)

                if st.session_state.get("confirm_action_token") == f"delete_sku_{sku_id}":
                    st.warning("本当に削除しますか？SKUと関連URLを削除します。")
                    dc1, dc2 = st.columns([1, 1], gap="small")
                    with dc1:
                        if st.button("削除確定", type="primary", key=f"sku_delete_confirm_{sku_id}"):
                            try:
                                delete_product(sku_id=sku_id)
                                st.session_state["confirm_action_token"] = None
                                st.success("SKUを削除しました。")
                                st.rerun()
                            except Exception as e:
                                st.session_state["confirm_action_token"] = None
                                st.error(f"削除に失敗しました: {e}")
                    with dc2:
                        if st.button("キャンセル", key=f"sku_delete_cancel_{sku_id}"):
                            st.session_state["confirm_action_token"] = None
                            st.rerun()

                # ★追加改善：SKU内の重複チェック
                urls = [_norm_str(a.get("url")) for a in aliases]
                alias_texts = [_norm_str(a.get("alias_text")) for a in aliases]
                dup_urls = _count_duplicates(urls)
                dup_texts = _count_duplicates(alias_texts)

                if dup_urls or dup_texts:
                    msg = "⚠️ SKU内で重複が見つかりました："
                    parts = []
                    if dup_urls:
                        parts.append(f"同URL {len(dup_urls)}種類")
                    if dup_texts:
                        parts.append(f"同alias_text {len(dup_texts)}種類")
                    st.warning(msg + " / ".join(parts))

                    with st.expander("重複の詳細（SKU内）", expanded=False):
                        if dup_urls:
                            st.markdown("**同URL（SKU内）**")
                            st.dataframe(
                                [{"url": u, "count": c} for u, c in sorted(dup_urls.items(), key=lambda x: (-x[1], x[0]))],
                                use_container_width=True,
                                hide_index=True,
                            )
                        if dup_texts:
                            st.markdown("**同alias_text（SKU内）**")
                            st.dataframe(
                                [{"alias_text": t, "count": c} for t, c in sorted(dup_texts.items(), key=lambda x: (-x[1], x[0]))],
                                use_container_width=True,
                                hide_index=True,
                            )

                c1, c2 = st.columns([1.2, 1], gap="large")
                with c1:
                    st.markdown("**SKU Info**")
                    st.code(
                        {
                            "sku_id": sku_id,
                            "display_name": p.get("display_name"),
                            "normalized_model": p.get("normalized_model"),
                            "variant": p.get("variant"),
                            "memory_gb": p.get("memory_gb"),
                            "perf_score": p.get("perf_score"),
                        }
                    )
                with c2:
                    st.markdown("**Aliases**")
                    if not aliases:
                        st.caption("alias はありません。")
                    else:
                        st.dataframe(
                            [
                                {
                                    "shop": a.get("shop"),
                                    "alias_text": a.get("alias_text"),
                                    "url": a.get("url"),
                                    "created_at": a.get("created_at"),
                                }
                                for a in aliases
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
