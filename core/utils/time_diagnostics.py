from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from typing import Any, Dict, Iterable, List, Sequence

from .memory_time import build_result_time_usage, classify_memory_time_match, primary_timestamp

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


DEFAULT_TIMEZONE = "Asia/Shanghai"

WEEKDAY_TEXT_TO_INDEX = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}

TOOL_TIME_RANGE_SPECS = {
    "just_now": {"intent_type": "刚才", "note": "近 2 小时窗口。"},
    "early_morning": {"intent_type": "凌晨", "note": "当天凌晨时间窗口。"},
    "today_morning": {"intent_type": "今天上午", "note": "当天上午时间窗口。"},
    "noon": {"intent_type": "中午", "note": "当天中午时间窗口。"},
    "today_afternoon": {"intent_type": "今天下午", "note": "当天下午时间窗口。"},
    "today_night": {"intent_type": "今晚", "note": "当天晚间时间窗口。"},
    "yesterday_early_morning": {"intent_type": "昨天凌晨", "note": "昨天凌晨时间窗口。"},
    "yesterday_morning": {"intent_type": "昨天上午", "note": "昨天上午时间窗口。"},
    "yesterday_noon": {"intent_type": "昨天中午", "note": "昨天中午时间窗口。"},
    "yesterday_afternoon": {"intent_type": "昨天下午", "note": "昨天下午时间窗口。"},
    "last_night": {"intent_type": "昨晚", "note": "昨天晚间时间窗口。"},
    "yesterday": {"intent_type": "昨天", "note": "昨天全天时间窗口。"},
    "a_few_days_ago": {"intent_type": "前几天", "note": "近 3 天时间窗口。"},
    "this_week": {"intent_type": "这周", "note": "本周时间窗口。"},
    "past_7_days": {"intent_type": "最近七天", "note": "近 7 天时间窗口。"},
    "last_weekend": {"intent_type": "上周末", "note": "上周末时间窗口。"},
    "this_month": {"intent_type": "这个月", "note": "本月时间窗口。"},
    "past_30_days": {"intent_type": "最近 30 天", "note": "近 30 天时间窗口。"},
    "past_3_months": {"intent_type": "最近 3 个月", "note": "近 3 个月时间窗口。"},
    "this_year": {"intent_type": "今年", "note": "今年时间窗口。"},
    "past_year": {"intent_type": "最近一年", "note": "近一年时间窗口。"},
}

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
        has_explicit_window = bool(start_ts > 0 and end_ts > 0 and end_ts >= start_ts)
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
            "has_explicit_window": has_explicit_window,
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


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def _time_window(
    start: datetime,
    end: datetime,
    intent_type: str,
    normalized_time_range: str,
    matched_phrases: Sequence[str],
    note: str,
    timezone_name: str,
) -> TimeIntentDiagnostic:
    return TimeIntentDiagnostic(
        matched=True,
        intent_type=intent_type,
        normalized_time_range=normalized_time_range,
        start_time=_format_datetime(start),
        end_time=_format_datetime(end),
        timezone=timezone_name,
        matched_phrases=list(matched_phrases or []),
        note=note,
    )


def _last_weekday_window(now: datetime, weekday: int) -> tuple[datetime, datetime]:
    current_week_start = _start_of_day(now) - timedelta(days=now.weekday())
    target_day = current_week_start - timedelta(days=7 - weekday)
    return _start_of_day(target_day), _end_of_day(target_day)


def tool_time_range_to_intent(
    time_range: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> TimeIntentDiagnostic:
    safe_range = str(time_range or "").strip()
    if not safe_range:
        return TimeIntentDiagnostic(timezone=timezone_name)

    now = _get_now(timezone_name)
    today = _start_of_day(now)
    yesterday = today - timedelta(days=1)

    if safe_range == "just_now":
        return _time_window(
            start=now - timedelta(hours=2),
            end=now,
            intent_type="刚才",
            normalized_time_range=safe_range,
            matched_phrases=[safe_range],
            note=TOOL_TIME_RANGE_SPECS[safe_range]["note"],
            timezone_name=timezone_name,
        )
    if safe_range == "early_morning":
        return _time_window(today, today.replace(hour=5, minute=59, second=59), "凌晨", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "today_morning":
        return _time_window(today.replace(hour=6), today.replace(hour=11, minute=59, second=59), "今天上午", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "noon":
        return _time_window(today.replace(hour=12), today.replace(hour=13, minute=59, second=59), "中午", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "today_afternoon":
        return _time_window(today.replace(hour=14), today.replace(hour=17, minute=59, second=59), "今天下午", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "today_night":
        return _time_window(today.replace(hour=18), _end_of_day(today), "今晚", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "yesterday_early_morning":
        return _time_window(yesterday, yesterday.replace(hour=5, minute=59, second=59), "昨天凌晨", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "yesterday_morning":
        return _time_window(yesterday.replace(hour=6), yesterday.replace(hour=11, minute=59, second=59), "昨天上午", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "yesterday_noon":
        return _time_window(yesterday.replace(hour=12), yesterday.replace(hour=13, minute=59, second=59), "昨天中午", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "yesterday_afternoon":
        return _time_window(yesterday.replace(hour=14), yesterday.replace(hour=17, minute=59, second=59), "昨天下午", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "last_night":
        return _time_window(yesterday.replace(hour=18), _end_of_day(yesterday), "昨晚", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "yesterday":
        return _time_window(_start_of_day(yesterday), _end_of_day(yesterday), "昨天", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "a_few_days_ago":
        return _time_window(_start_of_day(now - timedelta(days=3)), _end_of_day(now - timedelta(days=1)), "前几天", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "this_week":
        week_start = _start_of_day(now) - timedelta(days=now.weekday())
        return _time_window(week_start, _end_of_day(now), "这周", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "past_7_days":
        return _time_window(_start_of_day(now - timedelta(days=7)), _end_of_day(now), "最近七天", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "last_weekend":
        week_start = _start_of_day(now) - timedelta(days=now.weekday())
        saturday = week_start - timedelta(days=2)
        sunday = week_start - timedelta(days=1)
        return _time_window(_start_of_day(saturday), _end_of_day(sunday), "上周末", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "this_month":
        month_start = today.replace(day=1)
        return _time_window(month_start, _end_of_day(now), "这个月", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "past_30_days":
        return _time_window(_start_of_day(now - timedelta(days=30)), _end_of_day(now), "最近30天", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "past_3_months":
        return _time_window(_start_of_day(now - timedelta(days=90)), _end_of_day(now), "最近3个月", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "this_year":
        year_start = today.replace(month=1, day=1)
        return _time_window(year_start, _end_of_day(now), "今年", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)
    if safe_range == "past_year":
        return _time_window(_start_of_day(now - timedelta(days=365)), _end_of_day(now), "最近一年", safe_range, [safe_range], TOOL_TIME_RANGE_SPECS[safe_range]["note"], timezone_name)

    return TimeIntentDiagnostic(timezone=timezone_name)


def analyze_time_intent(
    text: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> TimeIntentDiagnostic:
    safe_text = str(text or "").strip()
    if not safe_text:
        return TimeIntentDiagnostic(timezone=timezone_name)

    now = _get_now(timezone_name)
    today = _start_of_day(now)
    yesterday = today - timedelta(days=1)

    tool_intent = tool_time_range_to_intent(safe_text, timezone_name=timezone_name)
    if tool_intent.matched:
        return tool_intent

    last_weekday_match = re.search(r"上周([一二三四五六日天])", safe_text)
    if last_weekday_match:
        weekday_text = last_weekday_match.group(1)
        weekday_index = WEEKDAY_TEXT_TO_INDEX.get(weekday_text)
        if weekday_index is not None:
            start, end = _last_weekday_window(now, weekday_index)
            return _time_window(
                start=start,
                end=end,
                intent_type=f"上周{weekday_text}",
                normalized_time_range=f"last_weekday_{weekday_index}",
                matched_phrases=[last_weekday_match.group(0)],
                note="显式上周星期表达，按上周对应自然日硬时间过滤。",
                timezone_name=timezone_name,
            )

    intent_specs = [
        {
            "intent_type": "今天上午",
            "normalized_time_range": "today_morning",
            "phrases": ["今天上午", "今早", "今天早上"],
            "start": today.replace(hour=6),
            "end": today.replace(hour=11, minute=59, second=59),
            "note": "映射到 today_morning。",
        },
        {
            "intent_type": "今天下午",
            "normalized_time_range": "today_afternoon",
            "phrases": ["今天下午"],
            "start": today.replace(hour=14),
            "end": today.replace(hour=17, minute=59, second=59),
            "note": "映射到 today_afternoon。",
        },
        {
            "intent_type": "今晚",
            "normalized_time_range": "today_night",
            "phrases": ["今晚", "今天晚上", "今夜"],
            "start": today.replace(hour=18),
            "end": _end_of_day(today),
            "note": "映射到 today_night。",
        },
        {
            "intent_type": "昨晚",
            "normalized_time_range": "last_night",
            "phrases": ["昨晚", "昨天晚上", "昨夜", "昨天夜里"],
            "start": yesterday.replace(hour=18),
            "end": _end_of_day(yesterday),
            "note": "映射到 last_night。",
        },
        {
            "intent_type": "昨天上午",
            "normalized_time_range": "yesterday_morning",
            "phrases": ["昨天上午", "昨天早上", "昨早"],
            "start": yesterday.replace(hour=6),
            "end": yesterday.replace(hour=11, minute=59, second=59),
            "note": "映射到 yesterday_morning。",
        },
        {
            "intent_type": "昨天下午",
            "normalized_time_range": "yesterday_afternoon",
            "phrases": ["昨天下午"],
            "start": yesterday.replace(hour=14),
            "end": yesterday.replace(hour=17, minute=59, second=59),
            "note": "映射到 yesterday_afternoon。",
        },
        {
            "intent_type": "昨天",
            "normalized_time_range": "yesterday",
            "phrases": ["昨天", "昨日"],
            "start": _start_of_day(yesterday),
            "end": _end_of_day(yesterday),
            "note": "映射到 yesterday。",
        },
        {
            "intent_type": "刚才",
            "normalized_time_range": "just_now",
            "phrases": ["刚才", "刚刚"],
            "start": now - timedelta(hours=2),
            "end": now,
            "note": "默认按近 2 小时窗口处理。",
        },
        {
            "intent_type": "前面",
            "normalized_time_range": "recent_context",
            "phrases": ["前面", "前文", "刚刚前面"],
            "start": now - timedelta(hours=6),
            "end": now,
            "note": "默认按近 6 小时当前会话窗口处理。",
        },
        {
            "intent_type": "之前",
            "normalized_time_range": "earlier_context",
            "phrases": ["之前", "那会儿", "当时"],
            "start": now - timedelta(days=7),
            "end": now,
            "note": "模糊历史指代，保守映射为近 7 天窗口。",
        },
        {
            "intent_type": "上次",
            "normalized_time_range": "last_time",
            "phrases": ["上次", "上回", "之前那次"],
            "start": now - timedelta(days=7),
            "end": now,
            "note": "模糊相对时间，保守映射为近 7 天窗口。",
        },
    ]

    for spec in intent_specs:
        matched_phrases = _match_phrases(safe_text, spec["phrases"])
        if not matched_phrases:
            continue
        return _time_window(
            start=spec["start"],
            end=spec["end"],
            intent_type=str(spec["intent_type"]),
            normalized_time_range=str(spec["normalized_time_range"]),
            matched_phrases=matched_phrases,
            note=str(spec["note"]),
            timezone_name=timezone_name,
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
    payload = intent.to_dict()
    payload["matched"] = bool(
        intent.matched
        and payload.get("has_explicit_window")
        and float(payload.get("end_ts", 0.0) or 0.0) >= float(payload.get("start_ts", 0.0) or 0.0)
    )
    return payload


def analyze_recall_request(text: str) -> Dict[str, Any]:
    safe_text = str(text or "").strip()
    if not safe_text:
        return {
            "matched": False,
            "matched_phrases": [],
            "raw_chat_priority": False,
            "recent_fact_priority": False,
            "time_expression_priority": False,
            "time_intent": TimeIntentDiagnostic().to_dict(),
        }

    matched_phrases = _match_phrases(safe_text, RECALL_TRIGGER_PHRASES)
    raw_chat_hits = _match_phrases(safe_text, RAW_CHAT_PRIORITY_PHRASES)
    time_intent = analyze_time_intent(safe_text)
    time_expression_priority = bool(time_intent.matched and time_intent.to_dict().get("has_explicit_window"))
    raw_chat_priority = bool(
        raw_chat_hits
        or time_intent.intent_type in {"刚才", "前面", "之前", "上次"}
    )
    matched = bool(matched_phrases or raw_chat_hits or time_intent.matched)
    return {
        "matched": matched,
        "matched_phrases": list(dict.fromkeys(matched_phrases + raw_chat_hits + list(time_intent.matched_phrases or []))),
        "raw_chat_priority": raw_chat_priority,
        "recent_fact_priority": bool(matched or time_expression_priority),
        "time_expression_priority": time_expression_priority,
        "time_intent": time_intent.to_dict(),
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
    rows: Sequence[Sequence[Any]],
    top_n: int = 5,
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
