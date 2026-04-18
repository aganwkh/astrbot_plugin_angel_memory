from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
from typing import Any, Dict, Iterable, List, Sequence

from .memory_time import build_result_time_usage, classify_memory_time_match, primary_timestamp

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


DEFAULT_TIMEZONE = "Asia/Shanghai"
TIME_SLOT_CONFIDENCE_THRESHOLD = 0.75
TIME_SLOT_CACHE_TTL_SECONDS = 600

CANONICAL_NON_TOOL_TIME_SLOT_INPUTS = {
    "recent_context": "\u524d\u9762",
    "earlier_context": "\u4e4b\u524d",
    "last_time": "\u4e0a\u6b21",
    "last_weekday_0": "\u4e0a\u5468\u4e00",
    "last_weekday_1": "\u4e0a\u5468\u4e8c",
    "last_weekday_2": "\u4e0a\u5468\u4e09",
    "last_weekday_3": "\u4e0a\u5468\u56db",
    "last_weekday_4": "\u4e0a\u5468\u4e94",
    "last_weekday_5": "\u4e0a\u5468\u516d",
    "last_weekday_6": "\u4e0a\u5468\u65e5",
}

LOW_INFORMATION_FOLLOWUP_PHRASES = [
    "\u90a3\u4e2a",
    "\u90a3\u6b21",
    "\u90a3\u6b21\u5462",
    "\u90a3\u4f1a\u513f",
    "\u5c31\u90a3\u4e2a",
    "\u5c31\u90a3\u6b21",
    "\u5c31\u90a3\u4e2a\u65f6\u95f4",
    "\u90fd\u804a\u4e86\u4e9b\u4ec0\u4e48",
    "\u804a\u4e86\u4e9b\u4ec0\u4e48",
    "\u90fd\u8bf4\u4e86\u4e9b\u4ec0\u4e48",
    "\u8bf4\u4e86\u4e9b\u4ec0\u4e48",
    "\u524d\u9762\u90a3\u4e2a\u5462",
    "\u4f60\u518d\u60f3\u60f3",
    "\u518d\u60f3\u60f3",
    "\u4e0d\u662f\u90a3\u4e2a\u5417",
    "\u4e0d\u662f\u90a3\u6b21\u5417",
    "\u5f53\u65f6\u90a3\u4e2a",
]

RECALL_REVIEW_HINT_PHRASES = [
    "\u8fd8\u8bb0\u5f97",
    "\u804a\u4e86\u4ec0\u4e48",
    "\u804a\u4e86\u4e9b\u4ec0\u4e48",
    "\u90fd\u804a\u4e86\u4e9b\u4ec0\u4e48",
    "\u8bf4\u4e86\u4ec0\u4e48",
    "\u8bf4\u4e86\u4e9b\u4ec0\u4e48",
    "\u90fd\u8bf4\u4e86\u4e9b\u4ec0\u4e48",
    "\u539f\u8bdd",
    "\u539f\u53e5",
    "\u63d0\u8fc7",
    "\u56de\u5fc6",
    "\u56de\u987e",
    "review",
    "recall",
]

TIME_SLOT_TRIGGER_TIME_PHRASES = [
    "\u4eca\u5929\u51cc\u6668",
    "\u51cc\u6668\u90a3\u4f1a\u513f",
    "\u6628\u5929\u51cc\u6668",
]

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
    "聊了些什么",
    "都聊了些什么",
    "聊过什么",
    "聊的什么",
    "聊了啥",
    "聊过啥",
    "说了什么",
    "说了些什么",
    "都说了些什么",
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
    "说了些什么",
    "都说了些什么",
    "说过什么",
    "说的什么",
    "说了啥",
    "聊了什么",
    "聊了些什么",
    "都聊了些什么",
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


@dataclass
class TimeSlotSelectionResult:
    decision: str = "abstain"
    selected_time_slot: str = ""
    confidence: float = 0.0
    reason: str = ""
    inherit_previous: bool = False
    abstain: bool = True
    parse_success: bool = False
    is_valid_slot: bool = False
    low_confidence: bool = False
    error: str = ""
    raw_response: str = ""
    raw_response_original: str = ""
    raw_response_sanitized: str = ""
    extraction_mode: str = ""
    parse_error: str = ""
    confidence_threshold: float = TIME_SLOT_CONFIDENCE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": str(self.decision or "abstain"),
            "selected_time_slot": str(self.selected_time_slot or ""),
            "confidence": float(self.confidence or 0.0),
            "reason": str(self.reason or ""),
            "inherit_previous": bool(self.inherit_previous),
            "abstain": bool(self.abstain),
            "parse_success": bool(self.parse_success),
            "is_valid_slot": bool(self.is_valid_slot),
            "low_confidence": bool(self.low_confidence),
            "error": str(self.error or ""),
            "raw_response": str(self.raw_response or ""),
            "raw_response_original": str(self.raw_response_original or ""),
            "raw_response_sanitized": str(self.raw_response_sanitized or ""),
            "extraction_mode": str(self.extraction_mode or ""),
            "parse_error": str(self.parse_error or ""),
            "confidence_threshold": float(self.confidence_threshold or 0.0),
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


def _strip_json_fence(text: str) -> str:
    safe_text = str(text or "").strip()
    if not safe_text.startswith("```"):
        return safe_text
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", safe_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return str(fenced_match.group(1) or "").strip()
    return safe_text


def _extract_first_json_object(text: str) -> str:
    safe_text = str(text or "")
    start_index = -1
    depth = 0
    in_string = False
    escaping = False
    for index, char in enumerate(safe_text):
        if start_index < 0:
            if char == "{":
                start_index = index
                depth = 1
            continue

        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return safe_text[start_index : index + 1].strip()
    return ""


def is_low_information_followup(text: str) -> bool:
    safe_text = re.sub(r"\s+", "", str(text or "").strip())
    if not safe_text:
        return False
    if safe_text in LOW_INFORMATION_FOLLOWUP_PHRASES:
        return True
    if len(safe_text) <= 8 and any(phrase in safe_text for phrase in LOW_INFORMATION_FOLLOWUP_PHRASES):
        return True
    return False


def is_recall_or_review_query(text: str, recall_request: Dict[str, Any] | None = None) -> bool:
    safe_text = str(text or "").strip()
    if not safe_text:
        return False
    recall_request = recall_request or {}
    if bool(recall_request.get("matched")):
        return True
    lowered = safe_text.lower()
    return any(phrase in lowered for phrase in RECALL_REVIEW_HINT_PHRASES)


def analyze_time_slot_classifier_trigger(
    text: str,
    recall_request: Dict[str, Any] | None = None,
    time_intent: TimeIntentDiagnostic | None = None,
) -> Dict[str, Any]:
    safe_text = str(text or "").strip()
    recall_request = recall_request or analyze_recall_request(safe_text)
    time_intent = time_intent or analyze_time_intent(safe_text)
    low_info_followup = is_low_information_followup(safe_text)
    recall_or_review = is_recall_or_review_query(safe_text, recall_request)
    contains_time_expression = bool(
        time_intent.matched or any(phrase in safe_text for phrase in TIME_SLOT_TRIGGER_TIME_PHRASES)
    )

    trigger_reasons: List[str] = []
    if contains_time_expression:
        trigger_reasons.append("contains_time_expression")
    if recall_or_review:
        trigger_reasons.append("recall_or_review_semantics")
    if low_info_followup:
        trigger_reasons.append("low_information_followup")

    return {
        "raw_user_input": safe_text,
        "contains_time_expression": contains_time_expression,
        "recall_or_review": recall_or_review,
        "low_information_followup": low_info_followup,
        "trigger_reason": trigger_reasons,
        "should_call_classifier": bool(trigger_reasons),
        "time_intent": time_intent.to_dict(),
        "recall_request": dict(recall_request or {}),
    }


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
            "intent_type": "凌晨",
            "normalized_time_range": "early_morning",
            "phrases": ["今天凌晨", "凌晨那会儿"],
            "start": today,
            "end": today.replace(hour=5, minute=59, second=59),
            "note": "映射到 early_morning。",
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
            "intent_type": "昨天凌晨",
            "normalized_time_range": "yesterday_early_morning",
            "phrases": ["昨天凌晨"],
            "start": yesterday,
            "end": yesterday.replace(hour=5, minute=59, second=59),
            "note": "映射到 yesterday_early_morning。",
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


def build_time_intent_from_slot(
    time_slot: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> TimeIntentDiagnostic:
    safe_slot = str(time_slot or "").strip()
    if not safe_slot:
        return TimeIntentDiagnostic(timezone=timezone_name)

    if safe_slot in TOOL_TIME_RANGE_SPECS:
        return tool_time_range_to_intent(safe_slot, timezone_name=timezone_name)

    canonical_input = CANONICAL_NON_TOOL_TIME_SLOT_INPUTS.get(safe_slot, "")
    if not canonical_input:
        return TimeIntentDiagnostic(timezone=timezone_name)

    intent = analyze_time_intent(canonical_input, timezone_name=timezone_name)
    if intent.matched and intent.normalized_time_range == safe_slot:
        return intent
    return TimeIntentDiagnostic(timezone=timezone_name)


def get_legal_time_slot_catalog(
    timezone_name: str = DEFAULT_TIMEZONE,
) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}

    for slot_name in TOOL_TIME_RANGE_SPECS.keys():
        intent = build_time_intent_from_slot(slot_name, timezone_name=timezone_name)
        if not intent.matched:
            continue
        payload = intent.to_dict()
        payload["slot_name"] = slot_name
        catalog[slot_name] = payload

    for slot_name in CANONICAL_NON_TOOL_TIME_SLOT_INPUTS.keys():
        intent = build_time_intent_from_slot(slot_name, timezone_name=timezone_name)
        if not intent.matched:
            continue
        payload = intent.to_dict()
        payload["slot_name"] = slot_name
        catalog[slot_name] = payload

    return catalog


def get_legal_time_slot_names(
    timezone_name: str = DEFAULT_TIMEZONE,
) -> List[str]:
    return sorted(get_legal_time_slot_catalog(timezone_name=timezone_name).keys())


def is_valid_legal_time_slot(
    time_slot: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> bool:
    safe_slot = str(time_slot or "").strip()
    if not safe_slot:
        return False
    intent = build_time_intent_from_slot(safe_slot, timezone_name=timezone_name)
    return bool(intent.matched and intent.normalized_time_range == safe_slot)


def _build_time_slot_semantic_definition(
    slot_name: str,
    payload: Dict[str, Any],
) -> str:
    intent_type = str(payload.get("intent_type", "") or "")
    note = str(payload.get("note", "") or "")
    start_time = str(payload.get("start_time", "") or "")
    end_time = str(payload.get("end_time", "") or "")

    if slot_name == "just_now":
        return "The very recent part of the current conversation context, anchored to immediacy rather than a calendar date."
    if slot_name == "recent_context":
        return "A nearby earlier part of the current conversation context, still recent but not necessarily the latest few messages."
    if slot_name == "earlier_context":
        return "Something mentioned earlier in the broader conversation context, less immediate than recent_context."
    if slot_name == "last_time":
        return "The previous occurrence or previous chat/topic turn, anchored to 'last time' semantics instead of a specific date."
    if slot_name.startswith("last_weekday_"):
        return (
            f"A specific weekday from last week. Current slot means {intent_type}. "
            f"Use it when the user clearly points to that exact weekday."
        )

    if start_time and end_time:
        return (
            f"{intent_type} semantic window. This slot covers the time span from "
            f"{start_time} to {end_time}. {note}"
        )
    return f"{intent_type} semantic window. {note}"


def _build_time_slot_representative_examples(slot_name: str) -> List[str]:
    direct_examples = {
        "just_now": ["刚才那个", "刚刚聊的", "前两句说的"],
        "recent_context": ["前面不是说过吗", "刚才前面那段", "前文提到的"],
        "earlier_context": ["之前那次", "更早前说过的", "前面某次聊到的"],
        "last_time": ["上次那个", "上回聊的", "前一次说过的"],
        "early_morning": ["今天凌晨那个", "今天半夜聊的", "凌晨一点那次"],
        "today_morning": ["今天上午那个", "早上聊的", "上午那会儿说的"],
        "noon": ["今天中午那个", "中午聊的", "午饭前后那次"],
        "today_afternoon": ["今天下午那个", "下午聊的", "下午那会儿说的"],
        "today_night": ["今晚那个", "今天晚上聊的", "今晚那次"],
        "yesterday_early_morning": ["昨天凌晨那个", "昨夜过零点那次", "昨天半夜聊的"],
        "yesterday_morning": ["昨天上午那个", "昨天早上聊的", "昨早那次"],
        "yesterday_noon": ["昨天中午那个", "昨天午饭前后", "昨儿中午聊的"],
        "yesterday_afternoon": ["昨天下午那个", "昨天下午聊的", "昨儿下午那次"],
        "last_night": ["昨晚那个", "昨天晚上聊的", "昨晚那次"],
        "yesterday": ["昨天那天", "昨天聊的", "前一天那次"],
        "a_few_days_ago": ["前几天那个", "这几天前面聊过的", "前几天那次"],
        "this_week": ["这周那个", "本周聊的", "这周里说过的"],
        "past_7_days": ["最近七天那个", "近一周聊的", "这七天里提过的"],
        "last_weekend": ["上周末那个", "上周六日聊的", "上个周末那次"],
        "this_month": ["这个月那个", "本月聊过的", "这个月里提到的"],
        "past_30_days": ["最近三十天那个", "近一个月聊的", "最近一个月提过的"],
        "past_3_months": ["最近三个月那个", "近三个月聊的", "这几个月里提过的"],
        "this_year": ["今年那个", "今年聊过的", "今年里提到的"],
        "past_year": ["最近一年那个", "近一年聊过的", "这一年里提过的"],
    }
    if slot_name in direct_examples:
        return direct_examples[slot_name]

    if slot_name.startswith("last_weekday_"):
        weekday_label = CANONICAL_NON_TOOL_TIME_SLOT_INPUTS.get(slot_name, "")
        return [
            f"{weekday_label}那个",
            f"{weekday_label}聊的",
            f"{weekday_label}那次",
        ]
    return []


def _build_time_slot_adjacent_boundaries(slot_name: str) -> List[str]:
    boundaries = {
        "just_now": [
            "Use this only for the immediately recent exchange, not for a broad earlier context.",
            "If the user means an older part of the same chat, prefer recent_context or earlier_context.",
        ],
        "recent_context": [
            "More recent than earlier_context, but broader than just_now.",
            "If the user clearly means the latest few turns, prefer just_now.",
        ],
        "earlier_context": [
            "Broader and less immediate than recent_context.",
            "If the user says '上次' or points to the previous occurrence, prefer last_time.",
        ],
        "last_time": [
            "Use this for '上次/上回/前一次' semantics, not generic earlier context.",
            "If the user points to a calendar day, prefer the explicit day slot instead.",
        ],
        "early_morning": [
            "Only for today after midnight and before the morning slot.",
            "If the user means last night or yesterday after midnight, prefer last_night or yesterday_early_morning.",
        ],
        "today_morning": [
            "Use this for today morning, not post-midnight pre-dawn or noon.",
            "If the user means lunch-time, prefer noon.",
        ],
        "noon": [
            "A narrow midday window, narrower than today_morning or today_afternoon.",
            "If the user only means 'today' without noon specificity, prefer the broader day slot only when available.",
        ],
        "today_afternoon": [
            "Use this after noon and before tonight.",
            "If the user means evening or night, prefer today_night.",
        ],
        "today_night": [
            "Use this for today's evening/night period.",
            "If the user means after midnight, prefer early_morning instead of today_night.",
        ],
        "yesterday_early_morning": [
            "This is yesterday after midnight and before yesterday morning.",
            "If the user means yesterday evening, prefer last_night.",
        ],
        "yesterday_morning": [
            "Use this for yesterday morning, not yesterday pre-dawn or noon.",
            "If the user only means yesterday without a daypart, the broader yesterday slot is safer.",
        ],
        "yesterday_noon": [
            "A narrow midday window inside yesterday.",
            "If the user means a broader yesterday period, prefer yesterday.",
        ],
        "yesterday_afternoon": [
            "Use this for yesterday afternoon, not yesterday night.",
            "If the user only says 昨天 with no finer clue, prefer yesterday.",
        ],
        "last_night": [
            "Use this for yesterday evening/night semantics.",
            "Do not use it for today after midnight; that belongs to early_morning.",
        ],
        "yesterday": [
            "Broad previous-day slot; if a narrower yesterday daypart is clearly expressed, prefer the narrower slot.",
            "Do not use when the user explicitly means last night only.",
        ],
        "a_few_days_ago": [
            "Broader than yesterday, but narrower than this_week or past_7_days when the user clearly says '前几天'.",
            "If the user identifies a precise weekday or weekend, prefer that more specific slot.",
        ],
        "this_week": [
            "Calendar-week semantics, not a rolling seven-day window.",
            "If the user means a rolling recent interval, prefer past_7_days.",
        ],
        "past_7_days": [
            "Rolling recent seven-day interval, not the calendar week.",
            "If the user explicitly says '这周', prefer this_week.",
        ],
        "last_weekend": [
            "Use only for last Saturday/Sunday semantics.",
            "If the user names a weekday from last week, prefer last_weekday_x.",
        ],
        "this_month": [
            "Calendar-month semantics.",
            "If the user means a rolling recent month, prefer past_30_days.",
        ],
        "past_30_days": [
            "Rolling recent 30-day interval.",
            "If the user explicitly says '这个月/本月', prefer this_month.",
        ],
        "past_3_months": [
            "Rolling recent three-month interval.",
            "If the user clearly points to this month only, prefer this_month.",
        ],
        "this_year": [
            "Calendar-year semantics.",
            "If the user means a rolling recent year, prefer past_year.",
        ],
        "past_year": [
            "Rolling recent one-year interval.",
            "If the user explicitly means this calendar year, prefer this_year.",
        ],
    }
    if slot_name in boundaries:
        return boundaries[slot_name]
    if slot_name.startswith("last_weekday_"):
        weekday_label = CANONICAL_NON_TOOL_TIME_SLOT_INPUTS.get(slot_name, "上周某天")
        return [
            f"Use this only when the user clearly points to {weekday_label}.",
            "If the user only says '上周末', prefer last_weekend; if the user says '上周' broadly, prefer a broader week slot when available.",
        ]
    return []


def get_time_slot_catalog_for_prompt(
    timezone_name: str = DEFAULT_TIMEZONE,
) -> List[Dict[str, Any]]:
    catalog = get_legal_time_slot_catalog(timezone_name=timezone_name)
    prompt_items: List[Dict[str, Any]] = []
    for slot_name in sorted(catalog.keys()):
        payload = catalog[slot_name]
        prompt_items.append(
            {
                "slot_name": slot_name,
                "intent_type": str(payload.get("intent_type", "") or ""),
                "note": str(payload.get("note", "") or ""),
                "start_time": str(payload.get("start_time", "") or ""),
                "end_time": str(payload.get("end_time", "") or ""),
                "semantic_definition": _build_time_slot_semantic_definition(slot_name, payload),
                "representative_examples": _build_time_slot_representative_examples(slot_name),
                "adjacent_boundaries": _build_time_slot_adjacent_boundaries(slot_name),
            }
        )
    return prompt_items


def parse_time_slot_selection_response(
    response_text: str,
    legal_slots: Sequence[str],
    confidence_threshold: float = TIME_SLOT_CONFIDENCE_THRESHOLD,
) -> TimeSlotSelectionResult:
    original_response = str(response_text or "")
    result = TimeSlotSelectionResult(
        raw_response=original_response,
        raw_response_original=original_response,
        confidence_threshold=float(confidence_threshold or 0.0),
    )
    safe_text = original_response.strip()
    result.raw_response_sanitized = safe_text
    if not safe_text:
        result.error = "empty_response"
        result.parse_error = "empty_response"
        result.extraction_mode = "empty"
        return result

    payload = None
    parse_attempts: List[str] = []
    extraction_candidates = [
        ("direct_json", safe_text),
        ("fenced_json", _strip_json_fence(safe_text)),
    ]
    embedded_source = _strip_json_fence(safe_text)
    extraction_candidates.append(("embedded_json", _extract_first_json_object(embedded_source)))

    tried_payloads = set()
    for extraction_mode, candidate_text in extraction_candidates:
        normalized_candidate = str(candidate_text or "").strip()
        if not normalized_candidate or normalized_candidate in tried_payloads:
            continue
        tried_payloads.add(normalized_candidate)
        result.raw_response_sanitized = normalized_candidate
        try:
            payload = json.loads(normalized_candidate)
            result.extraction_mode = extraction_mode
            result.parse_error = ""
            break
        except Exception as exc:
            parse_attempts.append(f"{extraction_mode}:{exc}")

    if payload is None:
        result.error = "json_parse_failed"
        result.parse_error = " | ".join(parse_attempts)
        if not result.extraction_mode:
            result.extraction_mode = "failed"
        return result

    if not isinstance(payload, dict):
        result.error = "payload_not_object"
        result.parse_error = "payload_not_object"
        return result

    if "decision" in payload:
        decision = payload.get("decision")
        selected_time_slot = payload.get("selected_time_slot", "")
        reason = payload.get("reason", "")
        confidence = payload.get("confidence")

        if not isinstance(decision, str):
            result.error = "invalid_decision_type"
            result.parse_error = "invalid_decision_type"
            return result
        if not isinstance(selected_time_slot, str):
            result.error = "invalid_selected_time_slot_type"
            result.parse_error = "invalid_selected_time_slot_type"
            return result
        if not isinstance(reason, str):
            result.error = "invalid_reason_type"
            result.parse_error = "invalid_reason_type"
            return result
        if confidence is not None and not isinstance(confidence, (int, float)):
            result.error = "invalid_confidence_type"
            result.parse_error = "invalid_confidence_type"
            return result

        normalized_decision = str(decision or "").strip()
        if normalized_decision not in {"selected_time_slot", "abstain", "inherit_previous"}:
            result.error = "invalid_decision_value"
            result.parse_error = "invalid_decision_value"
            return result

        result.parse_success = True
        result.decision = normalized_decision
        result.selected_time_slot = str(selected_time_slot or "").strip()
        result.reason = str(reason or "")
        result.confidence = float(
            confidence
            if confidence is not None
            else (1.0 if normalized_decision == "selected_time_slot" else 0.0)
        )
        result.inherit_previous = normalized_decision == "inherit_previous"
        result.abstain = normalized_decision == "abstain"
    else:
        required_fields = [
            "selected_time_slot",
            "confidence",
            "reason",
            "inherit_previous",
            "abstain",
        ]
        missing_fields = [field_name for field_name in required_fields if field_name not in payload]
        if missing_fields:
            result.error = f"missing_fields:{','.join(missing_fields)}"
            result.parse_error = result.error
            return result

        selected_time_slot = payload.get("selected_time_slot")
        confidence = payload.get("confidence")
        reason = payload.get("reason")
        inherit_previous = payload.get("inherit_previous")
        abstain = payload.get("abstain")

        if not isinstance(selected_time_slot, str):
            result.error = "invalid_selected_time_slot_type"
            result.parse_error = "invalid_selected_time_slot_type"
            return result
        if not isinstance(reason, str):
            result.error = "invalid_reason_type"
            result.parse_error = "invalid_reason_type"
            return result
        if not isinstance(inherit_previous, bool):
            result.error = "invalid_inherit_previous_type"
            result.parse_error = "invalid_inherit_previous_type"
            return result
        if not isinstance(abstain, bool):
            result.error = "invalid_abstain_type"
            result.parse_error = "invalid_abstain_type"
            return result
        if not isinstance(confidence, (int, float)):
            result.error = "invalid_confidence_type"
            result.parse_error = "invalid_confidence_type"
            return result

        result.parse_success = True
        result.decision = "inherit_previous" if bool(inherit_previous) else ("abstain" if bool(abstain) else "selected_time_slot")
        result.selected_time_slot = str(selected_time_slot or "").strip()
        result.confidence = float(confidence or 0.0)
        result.reason = str(reason or "")
        result.inherit_previous = bool(inherit_previous)
        result.abstain = bool(abstain)

    result.is_valid_slot = result.selected_time_slot in {str(slot) for slot in legal_slots}
    result.low_confidence = result.confidence < float(confidence_threshold or 0.0)

    if result.inherit_previous:
        return result
    if result.abstain:
        return result
    if not result.is_valid_slot:
        result.abstain = True
        result.error = "illegal_time_slot"
        result.parse_error = "illegal_time_slot"
        return result
    if result.low_confidence:
        result.abstain = True
        result.error = "low_confidence"
        result.parse_error = "low_confidence"
        return result
    return result


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
