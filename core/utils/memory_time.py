from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence


TIME_CONFIDENCE_EXACT = "exact"
TIME_CONFIDENCE_INFERRED = "inferred"
TIME_CONFIDENCE_LOW = "low_confidence"
TIME_CONFIDENCE_VALUES = {
    TIME_CONFIDENCE_EXACT,
    TIME_CONFIDENCE_INFERRED,
    TIME_CONFIDENCE_LOW,
}

_WEEKDAY_TEXT_TO_INDEX = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return float(default)


def normalize_time_confidence(value: Any) -> str:
    safe_value = str(value or "").strip().lower()
    if safe_value in TIME_CONFIDENCE_VALUES:
        return safe_value
    if safe_value in {"high", "precise"}:
        return TIME_CONFIDENCE_EXACT
    if safe_value in {"low", "low_conf"}:
        return TIME_CONFIDENCE_LOW
    if safe_value:
        return TIME_CONFIDENCE_INFERRED
    return TIME_CONFIDENCE_LOW


def normalize_source_message_ids(value: Any) -> List[str]:
    raw_items: List[Any] = []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raw_items = []
        else:
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, TypeError, ValueError):
                raw_items = [part.strip() for part in stripped.split(",")]
            else:
                raw_items = parsed if isinstance(parsed, list) else [stripped]
    elif value is None:
        raw_items = []
    else:
        raw_items = [value]

    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        message_id = str(item or "").strip()
        if not message_id or message_id in seen:
            continue
        seen.add(message_id)
        normalized.append(message_id)
    return normalized


def serialize_source_message_ids(value: Any) -> str:
    return json.dumps(normalize_source_message_ids(value), ensure_ascii=False)


def extract_chat_record_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = str(item.get("text", "") or "").strip()
                    if text:
                        parts.append(text)
                else:
                    text = str(item.get("text", "") or item.get("content", "") or "").strip()
                    if text:
                        parts.append(text)
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return " ".join(parts).strip()
    if isinstance(content, dict):
        return str(content.get("text", "") or content.get("content", "") or "").strip()
    return ""


def build_source_message_id(record: Dict[str, Any]) -> str:
    for key in ("message_id", "msg_id", "id", "record_id", "tool_call_id"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value

    sender_id = str(record.get("sender_id", "") or "").strip() or "unknown"
    timestamp = safe_float(record.get("timestamp", 0.0))
    content = extract_chat_record_text(record.get("content", ""))
    digest = hashlib.sha1(
        f"{sender_id}|{timestamp:.6f}|{content}".encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    return f"chat_{digest}"


def derive_time_metadata_from_chat_records(
    chat_records: Sequence[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    source_ids: List[str] = []
    timestamps: List[float] = []
    for record in chat_records or []:
        if not isinstance(record, dict):
            continue
        source_ids.append(build_source_message_id(record))
        timestamp = safe_float(record.get("timestamp", 0.0))
        if timestamp > 0:
            timestamps.append(timestamp)

    source_message_ids = normalize_source_message_ids(source_ids)
    if not timestamps:
        return {
            "source_message_ids": source_message_ids,
            "source_start_ts": 0.0,
            "source_end_ts": 0.0,
            "event_start_ts": 0.0,
            "event_end_ts": 0.0,
            "event_time_confidence": TIME_CONFIDENCE_LOW,
        }

    start_ts = min(timestamps)
    end_ts = max(timestamps)
    return {
        "source_message_ids": source_message_ids,
        "source_start_ts": start_ts,
        "source_end_ts": end_ts,
        "event_start_ts": start_ts,
        "event_end_ts": end_ts,
        "event_time_confidence": TIME_CONFIDENCE_EXACT,
    }


def infer_event_time_from_text(
    text: str,
    reference_ts: float,
) -> Dict[str, Any]:
    safe_text = str(text or "").strip()
    reference = safe_float(reference_ts)
    if not safe_text or reference <= 0:
        return {}

    now = datetime.fromtimestamp(reference)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)

    explicit_specs = [
        ("今天上午", today.replace(hour=6), today.replace(hour=11, minute=59, second=59)),
        ("今天下午", today.replace(hour=14), today.replace(hour=17, minute=59, second=59)),
        ("今晚", today.replace(hour=18), today.replace(hour=23, minute=59, second=59)),
        ("今天晚上", today.replace(hour=18), today.replace(hour=23, minute=59, second=59)),
        ("今夜", today.replace(hour=18), today.replace(hour=23, minute=59, second=59)),
        ("昨晚", yesterday.replace(hour=18), yesterday.replace(hour=23, minute=59, second=59)),
        ("昨天晚上", yesterday.replace(hour=18), yesterday.replace(hour=23, minute=59, second=59)),
        ("昨夜", yesterday.replace(hour=18), yesterday.replace(hour=23, minute=59, second=59)),
        ("昨天上午", yesterday.replace(hour=6), yesterday.replace(hour=11, minute=59, second=59)),
        ("昨早", yesterday.replace(hour=6), yesterday.replace(hour=11, minute=59, second=59)),
        ("昨天", yesterday.replace(hour=0), yesterday.replace(hour=23, minute=59, second=59)),
    ]
    for phrase, start_dt, end_dt in explicit_specs:
        if phrase in safe_text:
            return {
                "event_start_ts": start_dt.timestamp(),
                "event_end_ts": end_dt.timestamp(),
                "event_time_confidence": TIME_CONFIDENCE_LOW,
            }

    weekday_match = re.search(r"上周([一二三四五六日天])", safe_text)
    if weekday_match:
        weekday_text = weekday_match.group(1)
        weekday_index = _WEEKDAY_TEXT_TO_INDEX.get(weekday_text)
        if weekday_index is not None:
            current_week_start = today - timedelta(days=today.weekday())
            target_day = current_week_start - timedelta(days=7 - weekday_index)
            start_dt = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = target_day.replace(hour=23, minute=59, second=59, microsecond=0)
            return {
                "event_start_ts": start_dt.timestamp(),
                "event_end_ts": end_dt.timestamp(),
                "event_time_confidence": TIME_CONFIDENCE_LOW,
            }
    return {}


def apply_time_metadata_defaults(
    time_metadata: Optional[Dict[str, Any]],
    *,
    created_at: float = 0.0,
    text_for_inference: str = "",
) -> Dict[str, Any]:
    metadata = dict(time_metadata or {})
    metadata["source_message_ids"] = normalize_source_message_ids(
        metadata.get("source_message_ids", [])
    )
    metadata["source_start_ts"] = safe_float(metadata.get("source_start_ts", 0.0))
    metadata["source_end_ts"] = safe_float(metadata.get("source_end_ts", 0.0))
    metadata["event_start_ts"] = safe_float(metadata.get("event_start_ts", 0.0))
    metadata["event_end_ts"] = safe_float(metadata.get("event_end_ts", 0.0))
    metadata["event_time_confidence"] = normalize_time_confidence(
        metadata.get("event_time_confidence", "")
    )

    if metadata["source_start_ts"] > 0 and metadata["source_end_ts"] <= 0:
        metadata["source_end_ts"] = metadata["source_start_ts"]
    if metadata["source_end_ts"] > 0 and metadata["source_start_ts"] <= 0:
        metadata["source_start_ts"] = metadata["source_end_ts"]

    if metadata["event_start_ts"] > 0 and metadata["event_end_ts"] <= 0:
        metadata["event_end_ts"] = metadata["event_start_ts"]
    if metadata["event_end_ts"] > 0 and metadata["event_start_ts"] <= 0:
        metadata["event_start_ts"] = metadata["event_end_ts"]

    if metadata["event_start_ts"] <= 0 and metadata["source_start_ts"] > 0:
        metadata["event_start_ts"] = metadata["source_start_ts"]
        metadata["event_end_ts"] = metadata["source_end_ts"]
        if metadata["event_time_confidence"] == TIME_CONFIDENCE_LOW:
            metadata["event_time_confidence"] = TIME_CONFIDENCE_EXACT

    if metadata["event_start_ts"] <= 0:
        inferred = infer_event_time_from_text(text_for_inference, created_at)
        if inferred:
            metadata["event_start_ts"] = safe_float(inferred.get("event_start_ts", 0.0))
            metadata["event_end_ts"] = safe_float(inferred.get("event_end_ts", 0.0))
            metadata["event_time_confidence"] = normalize_time_confidence(
                inferred.get("event_time_confidence", TIME_CONFIDENCE_LOW)
            )

    return metadata


def get_memory_time_fields(memory: Any) -> Dict[str, Any]:
    created_at = safe_float(getattr(memory, "created_at", 0.0))
    metadata = apply_time_metadata_defaults(
        {
            "source_message_ids": getattr(memory, "source_message_ids", []),
            "source_start_ts": getattr(memory, "source_start_ts", 0.0),
            "source_end_ts": getattr(memory, "source_end_ts", 0.0),
            "event_start_ts": getattr(memory, "event_start_ts", 0.0),
            "event_end_ts": getattr(memory, "event_end_ts", 0.0),
            "event_time_confidence": getattr(memory, "event_time_confidence", ""),
        },
        created_at=created_at,
        text_for_inference="\n".join(
            [
                str(getattr(memory, "judgment", "") or "").strip(),
                str(getattr(memory, "reasoning", "") or "").strip(),
            ]
        ).strip(),
    )
    metadata["created_at"] = created_at
    metadata["has_source_time"] = bool(
        metadata["source_start_ts"] > 0 and metadata["source_end_ts"] >= metadata["source_start_ts"]
    )
    metadata["has_event_time"] = bool(
        metadata["event_start_ts"] > 0 and metadata["event_end_ts"] >= metadata["event_start_ts"]
    )
    metadata["has_created_at"] = bool(created_at > 0)
    return metadata


def overlaps_time_window(
    start_ts: float,
    end_ts: float,
    window_start_ts: float,
    window_end_ts: float,
) -> bool:
    start = safe_float(start_ts)
    end = safe_float(end_ts)
    window_start = safe_float(window_start_ts)
    window_end = safe_float(window_end_ts)
    if start <= 0 or end < start or window_start <= 0 or window_end < window_start:
        return False
    return not (end < window_start or start > window_end)


def classify_memory_time_match(
    memory: Any,
    time_filter: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    fields = get_memory_time_fields(memory)
    if not isinstance(time_filter, dict):
        return {
            **fields,
            "matched": False,
            "match_type": "none",
            "primary_timestamp_field": primary_timestamp_field(memory),
        }

    start_ts = safe_float(time_filter.get("start_ts", 0.0))
    end_ts = safe_float(time_filter.get("end_ts", 0.0))
    if start_ts <= 0 or end_ts < start_ts:
        return {
            **fields,
            "matched": False,
            "match_type": "none",
            "primary_timestamp_field": primary_timestamp_field(memory),
        }

    event_match = overlaps_time_window(
        fields["event_start_ts"],
        fields["event_end_ts"],
        start_ts,
        end_ts,
    )
    source_match = overlaps_time_window(
        fields["source_start_ts"],
        fields["source_end_ts"],
        start_ts,
        end_ts,
    )
    created_at_match = start_ts <= fields["created_at"] <= end_ts if fields["created_at"] > 0 else False

    match_type = "none"
    if event_match:
        if fields["event_time_confidence"] == TIME_CONFIDENCE_EXACT:
            match_type = "event_exact"
        else:
            match_type = "event_inferred"
    elif source_match:
        match_type = "source_exact"
    elif created_at_match:
        match_type = "created_at_only"

    return {
        **fields,
        "matched": match_type != "none",
        "match_type": match_type,
        "primary_timestamp_field": primary_timestamp_field(memory),
    }


def primary_timestamp(memory: Any) -> float:
    fields = get_memory_time_fields(memory)
    for key in ("event_end_ts", "source_end_ts", "created_at"):
        timestamp = safe_float(fields.get(key, 0.0))
        if timestamp > 0:
            return timestamp
    return 0.0


def primary_timestamp_field(memory: Any) -> str:
    fields = get_memory_time_fields(memory)
    if safe_float(fields.get("event_end_ts", 0.0)) > 0:
        return "event_end_ts"
    if safe_float(fields.get("source_end_ts", 0.0)) > 0:
        return "source_end_ts"
    if safe_float(fields.get("created_at", 0.0)) > 0:
        return "created_at"
    return ""


def time_sort_boost(
    memory: Any,
    time_filter: Optional[Dict[str, Any]],
) -> float:
    if not isinstance(time_filter, dict):
        return 0.0

    match = classify_memory_time_match(memory, time_filter)
    match_type = match.get("match_type", "none")
    if match_type == "event_exact":
        return 1.0
    if match_type == "event_inferred":
        return 0.75
    if match_type == "source_exact":
        return 0.65
    if match_type == "created_at_only":
        return 0.2
    return 0.0


def build_result_time_usage(memories: Sequence[Any], time_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    exact_event_time = 0
    inferred_event_time = 0
    source_time_only = 0
    fallback_created_at_only = 0
    missing_event_time = 0
    missing_source_time = 0
    primary_field_counts: Dict[str, int] = {}

    for memory in memories or []:
        match = classify_memory_time_match(memory, time_filter)
        if not match.get("has_event_time"):
            missing_event_time += 1
        if not match.get("has_source_time"):
            missing_source_time += 1

        match_type = str(match.get("match_type", "none") or "none")
        if match_type == "event_exact":
            exact_event_time += 1
        elif match_type == "event_inferred":
            inferred_event_time += 1
        elif match_type == "source_exact":
            source_time_only += 1
        elif match_type == "created_at_only":
            fallback_created_at_only += 1

        primary_field = str(match.get("primary_timestamp_field", "") or "")
        primary_field_counts[primary_field] = primary_field_counts.get(primary_field, 0) + 1

    return {
        "exact_event_time": exact_event_time,
        "inferred_event_time": inferred_event_time,
        "source_time_only": source_time_only,
        "fallback_created_at_only": fallback_created_at_only,
        "missing_event_time": missing_event_time,
        "missing_source_time": missing_source_time,
        "primary_timestamp_field_counts": primary_field_counts,
    }


def summarize_memories_by_time_field(
    memories: Sequence[Any],
    *,
    field: str = "event",
    top_n: int = 5,
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    day_buckets: Dict[str, int] = {}
    timestamps: List[float] = []
    missing_timestamp = 0

    for memory in memories or []:
        match = classify_memory_time_match(memory, None)
        if field == "created_at":
            timestamp = safe_float(getattr(memory, "created_at", 0.0))
            timestamp_field = "created_at"
            note = "长期记忆写入时间分布基于 created_at，仅用于调试与辅助排序。"
        else:
            timestamp = primary_timestamp(memory)
            timestamp_field = "event_end_ts/source_end_ts"
            note = "长期记忆结果优先基于 event_end_ts，其次 source_end_ts 汇总。"

        if timestamp <= 0:
            missing_timestamp += 1
            formatted_time = ""
        else:
            formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            day_key = formatted_time[:10]
            day_buckets[day_key] = day_buckets.get(day_key, 0) + 1
            timestamps.append(timestamp)

        samples.append(
            {
                "id": str(getattr(memory, "id", "") or ""),
                "timestamp": timestamp,
                "time": formatted_time,
                "preview": str(getattr(memory, "judgment", "") or "")[:80],
                "score": getattr(memory, "similarity", 0.0),
                "primary_timestamp_field": match.get("primary_timestamp_field", ""),
                "event_time_confidence": match.get("event_time_confidence", ""),
            }
        )

    samples.sort(key=lambda item: float(item.get("timestamp", 0.0) or 0.0), reverse=True)
    return {
        "count": len(samples),
        "timestamp_field": timestamp_field,
        "earliest": datetime.fromtimestamp(min(timestamps)).strftime("%Y-%m-%d %H:%M:%S")
        if timestamps
        else "",
        "latest": datetime.fromtimestamp(max(timestamps)).strftime("%Y-%m-%d %H:%M:%S")
        if timestamps
        else "",
        "day_buckets": [
            {"day": day, "count": count}
            for day, count in sorted(day_buckets.items(), reverse=True)
        ],
        "missing_timestamp": missing_timestamp,
        "samples": samples[: max(0, int(top_n))],
        "note": note,
    }
