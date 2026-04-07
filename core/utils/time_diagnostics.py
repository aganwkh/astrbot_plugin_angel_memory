from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from typing import Any, Dict, Iterable, List, Sequence

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


DEFAULT_TIMEZONE = "Asia/Shanghai"

RECALL_TRIGGER_PHRASES = [
    "聊了什么",
    "聊过什么",
    "聊的什么",
    "聊了啥",
    "聊过啥",
    "说了什么",
    "说过什么",
    "说的什么",
    "说了啥",
    "说过啥",
    "原话是什么",
    "原话",
    "原句",
    "提过",
    "提到过",
    "聊过",
    "还记得",
    "是不是聊过",
    "是不是提过",
    "有没有聊过",
    "有没有提过",
]

RAW_CHAT_PRIORITY_PHRASES = [
    "原话",
    "原句",
    "说了什么",
    "说过什么",
    "说的什么",
    "说了啥",
    "聊了什么",
    "聊过什么",
    "聊的什么",
    "聊了啥",
    "提过",
    "刚才",
    "刚刚",
    "前面",
    "前文",
]


@dataclass
class TimeIntentDiagnostic:
    matched: bool = False
    intent_type: str = ""
    normalized_time_range: str = ""
    start_time: str = ""
    end_time: str = ""
    timezone: str = DEFAULT_TIMEZONE
    matched_phrases: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        start_ts = 0.0
        end_ts = 0.0
        start_dt = _parse_datetime(self.start_time, self.timezone)
        end_dt = _parse_datetime(self.end_time, self.timezone)
        if start_dt is not None:
            start_ts = float(start_dt.timestamp())
        if end_dt is not None:
            end_ts = float(end_dt.timestamp())
        return {
            "matched": bool(self.matched),
            "intent_type": str(self.intent_type or ""),
            "normalized_time_range": str(self.normalized_time_range or ""),
            "start_time": str(self.start_time or ""),
            "end_time": str(self.end_time or ""),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "timezone": str(self.timezone or DEFAULT_TIMEZONE),
            "matched_phrases": list(self.matched_phrases or []),
            "note": str(self.note or ""),
        }


def _get_now(timezone_name: str = DEFAULT_TIMEZONE) -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(timezone_name))
        except Exception:
            pass
    return datetime.now()


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _parse_datetime(value: str, timezone_name: str = DEFAULT_TIMEZONE) -> datetime | None:
    safe_value = str(value or "").strip()
    if not safe_value:
        return None
    try:
        parsed = datetime.strptime(safe_value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

    if ZoneInfo is not None:
        try:
            return parsed.replace(tzinfo=ZoneInfo(timezone_name))
        except Exception:
            return parsed
    return parsed


def _match_phrases(text: str, phrases: Sequence[str]) -> List[str]:
    safe_text = str(text or "")
    matched: List[str] = []
    for phrase in phrases:
        if not phrase:
            continue
        if re.search(re.escape(phrase), safe_text, flags=re.IGNORECASE):
            matched.append(phrase)
    return matched


def analyze_time_intent(
    text: str, timezone_name: str = DEFAULT_TIMEZONE
) -> TimeIntentDiagnostic:
    safe_text = str(text or "").strip()
    if not safe_text:
        return TimeIntentDiagnostic(timezone=timezone_name)

    now = _get_now(timezone_name)
    yesterday = now - timedelta(days=1)

    intent_specs = [
        {
            "intent_type": "昨晚",
            "normalized_time_range": "last_night",
            "phrases": ["昨晚", "昨天晚上", "昨夜", "昨天夜里"],
            "start": yesterday.replace(hour=18, minute=0, second=0, microsecond=0),
            "end": yesterday.replace(hour=23, minute=59, second=59, microsecond=0),
            "note": "可映射到现有 time_range=last_night。",
        },
        {
            "intent_type": "昨天",
            "normalized_time_range": "yesterday",
            "phrases": ["昨天", "昨日"],
            "start": yesterday.replace(hour=0, minute=0, second=0, microsecond=0),
            "end": yesterday.replace(hour=23, minute=59, second=59, microsecond=0),
            "note": "可映射到现有 time_range=yesterday。",
        },
        {
            "intent_type": "刚才",
            "normalized_time_range": "just_now",
            "phrases": ["刚才", "刚刚"],
            "start": now - timedelta(hours=2),
            "end": now,
            "note": "按现有 recall_by_time 逻辑，窗口近似为最近 2 小时。",
        },
        {
            "intent_type": "前面",
            "normalized_time_range": "recent_context",
            "phrases": ["前面", "前文", "前一段", "刚刚前面"],
            "start": now - timedelta(hours=6),
            "end": now,
            "note": "默认映射为当前会话近 6 小时窗口，并优先参考 raw_chat_window。",
        },
        {
            "intent_type": "上次",
            "normalized_time_range": "",
            "phrases": ["上次", "上回", "之前那次"],
            "start": None,
            "end": None,
            "note": "当前代码库没有为“上次”定义统一时间窗口。",
        },
    ]

    for spec in intent_specs:
        matched_phrases = _match_phrases(safe_text, spec["phrases"])
        if not matched_phrases:
            continue

        start = spec.get("start")
        end = spec.get("end")
        return TimeIntentDiagnostic(
            matched=True,
            intent_type=str(spec["intent_type"]),
            normalized_time_range=str(spec["normalized_time_range"]),
            start_time=_format_datetime(start) if isinstance(start, datetime) else "",
            end_time=_format_datetime(end) if isinstance(end, datetime) else "",
            timezone=timezone_name,
            matched_phrases=matched_phrases,
            note=str(spec.get("note", "")),
        )

    return TimeIntentDiagnostic(timezone=timezone_name)


def compare_time_intent(before_text: str, after_text: str) -> Dict[str, Any]:
    before = analyze_time_intent(before_text)
    after = analyze_time_intent(after_text)
    return {
        "before_matched": before.matched,
        "before_intent_type": before.intent_type,
        "before_normalized_time_range": before.normalized_time_range,
        "before_matched_phrases": before.matched_phrases,
        "after_matched": after.matched,
        "after_intent_type": after.intent_type,
        "after_normalized_time_range": after.normalized_time_range,
        "after_matched_phrases": after.matched_phrases,
        "lost": bool(before.matched and not after.matched),
    }


def build_time_filter_payload(
    text_or_intent: str | TimeIntentDiagnostic,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> Dict[str, Any]:
    intent = (
        text_or_intent
        if isinstance(text_or_intent, TimeIntentDiagnostic)
        else analyze_time_intent(str(text_or_intent or ""), timezone_name=timezone_name)
    )
    start_dt = _parse_datetime(intent.start_time, intent.timezone or timezone_name)
    end_dt = _parse_datetime(intent.end_time, intent.timezone or timezone_name)
    start_ts = float(start_dt.timestamp()) if start_dt is not None else 0.0
    end_ts = float(end_dt.timestamp()) if end_dt is not None else 0.0
    has_explicit_window = bool(start_ts > 0 and end_ts > 0 and end_ts >= start_ts)
    return {
        "matched": bool(intent.matched and has_explicit_window),
        "intent_type": str(intent.intent_type or ""),
        "normalized_time_range": str(intent.normalized_time_range or ""),
        "start_time": str(intent.start_time or ""),
        "end_time": str(intent.end_time or ""),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "timezone": str(intent.timezone or timezone_name or DEFAULT_TIMEZONE),
        "matched_phrases": list(intent.matched_phrases or []),
        "note": str(intent.note or ""),
        "has_explicit_window": has_explicit_window,
    }


def analyze_recall_request(text: str) -> Dict[str, Any]:
    safe_text = str(text or "").strip()
    if not safe_text:
        return {
            "matched": False,
            "matched_phrases": [],
            "raw_chat_priority": False,
            "recent_fact_priority": False,
        }

    matched_phrases = _match_phrases(safe_text, RECALL_TRIGGER_PHRASES)
    raw_chat_hits = _match_phrases(safe_text, RAW_CHAT_PRIORITY_PHRASES)
    matched = bool(matched_phrases or raw_chat_hits)
    return {
        "matched": matched,
        "matched_phrases": list(dict.fromkeys(matched_phrases + raw_chat_hits)),
        "raw_chat_priority": bool(raw_chat_hits),
        "recent_fact_priority": matched,
    }


def preview_text(text: str, limit: int = 120) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if len(normalized) <= limit:
        return normalized
    if limit <= 3:
        return normalized[:limit]
    return f"{normalized[: limit - 3]}..."


def get_event_diagnostic_store(event: Any) -> Dict[str, Any]:
    store = getattr(event, "_angel_memory_diagnostics", None)
    if isinstance(store, dict):
        return store
    store = {}
    setattr(event, "_angel_memory_diagnostics", store)
    return store


def _summarize_timestamped_items(
    items: Iterable[Dict[str, Any]],
    timestamp_field: str,
    note: str = "",
    top_n: int = 5,
) -> Dict[str, Any]:
    normalized: List[Dict[str, Any]] = []
    missing_timestamp = 0
    for item in items:
        try:
            ts = float(item.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts <= 0:
            missing_timestamp += 1
        normalized.append(
            {
                "id": str(item.get("id", "") or ""),
                "timestamp": ts,
                "time": _format_datetime(datetime.fromtimestamp(ts)) if ts > 0 else "",
                "preview": preview_text(str(item.get("preview", "") or ""), 80),
                "score": item.get("score"),
                "source": str(item.get("source", "") or ""),
            }
        )

    normalized.sort(key=lambda x: float(x.get("timestamp", 0.0) or 0.0), reverse=True)
    day_buckets: Dict[str, int] = {}
    for item in normalized:
        if not item["time"]:
            continue
        day_key = item["time"][:10]
        day_buckets[day_key] = day_buckets.get(day_key, 0) + 1

    timestamps = [item["timestamp"] for item in normalized if float(item["timestamp"]) > 0]
    return {
        "count": len(normalized),
        "timestamp_field": timestamp_field,
        "earliest": _format_datetime(datetime.fromtimestamp(min(timestamps))) if timestamps else "",
        "latest": _format_datetime(datetime.fromtimestamp(max(timestamps))) if timestamps else "",
        "day_buckets": [
            {"day": day, "count": count}
            for day, count in sorted(day_buckets.items(), reverse=True)
        ],
        "missing_timestamp": missing_timestamp,
        "samples": normalized[: max(0, int(top_n))],
        "note": str(note or ""),
    }


def summarize_memory_records(memories: Sequence[Any], top_n: int = 5) -> Dict[str, Any]:
    items = []
    for memory in memories or []:
        items.append(
            {
                "id": str(getattr(memory, "id", "") or ""),
                "timestamp": getattr(memory, "created_at", 0.0),
                "preview": getattr(memory, "judgment", "") or "",
                "score": getattr(memory, "similarity", 0.0),
                "source": "long_term_memory",
            }
        )
    return _summarize_timestamped_items(
        items=items,
        timestamp_field="created_at",
        note="长期记忆时间分布基于 BaseMemory.created_at。",
        top_n=top_n,
    )


def summarize_note_records(notes: Sequence[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    items = []
    for note in notes or []:
        metadata = note.get("metadata", {}) if isinstance(note, dict) else {}
        items.append(
            {
                "id": str(note.get("id", "") or ""),
                "timestamp": metadata.get("updated_at", 0.0),
                "preview": note.get("content", "") or "",
                "score": note.get("similarity", 0.0),
                "source": "note_index",
            }
        )
    return _summarize_timestamped_items(
        items=items,
        timestamp_field="updated_at",
        note="笔记时间分布基于 note metadata.updated_at。",
        top_n=top_n,
    )


def summarize_session_memories(memories: Sequence[Any], top_n: int = 5) -> Dict[str, Any]:
    items = []
    for memory in memories or []:
        items.append(
            {
                "id": str(getattr(memory, "id", "") or ""),
                "timestamp": getattr(memory, "created_at", 0.0),
                "preview": getattr(memory, "judgment", "") or "",
                "score": getattr(memory, "life_points", 0),
                "source": "session_memory_pool",
            }
        )
    return _summarize_timestamped_items(
        items=items,
        timestamp_field="session_memory.created_at",
        note="会话短时记忆时间分布基于加入 session 池的时间，不是长期记忆原始 created_at。",
        top_n=top_n,
    )


def summarize_raw_chat_rows(
    rows: Sequence[Sequence[Any]], top_n: int = 5
) -> Dict[str, Any]:
    items = []
    for row in rows or []:
        role = str(row[0] if len(row) > 0 else "" or "")
        content = str(row[1] if len(row) > 1 else "" or "")
        timestamp = row[2] if len(row) > 2 else 0.0
        items.append(
            {
                "id": role,
                "timestamp": timestamp,
                "preview": f"{role}: {content}",
                "score": "",
                "source": "raw_chat_window",
            }
        )
    return _summarize_timestamped_items(
        items=items,
        timestamp_field="raw_chat_window.timestamp",
        note="原始聊天窗口时间分布基于 raw_chat_window.db 中的 timestamp。",
        top_n=top_n,
    )
