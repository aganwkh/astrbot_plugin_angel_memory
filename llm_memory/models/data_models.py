"""
LLM 记忆系统的数据模型。
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import List, Optional

from ...core.utils.memory_time import (
    TIME_CONFIDENCE_LOW,
    apply_time_metadata_defaults,
    normalize_source_message_ids,
)
from ..config.system_config import KNOWLEDGE_CORE_SEPARATOR


class MemoryType(Enum):
    """记忆类型。"""

    KNOWLEDGE = "知识记忆"
    EVENT = "事件记忆"
    SKILL = "技能记忆"
    TASK = "任务记忆"
    EMOTIONAL = "情感记忆"


class MemoryError(Exception):
    """记忆系统基础异常。"""


class VectorizationError(MemoryError):
    """向量化相关异常。"""


class StorageError(MemoryError):
    """存储与检索相关异常。"""


class ValidationError(MemoryError):
    """数据校验异常。"""


class BaseMemory:
    """统一三元组记忆模型。"""

    def __init__(
        self,
        memory_type: MemoryType,
        judgment: str,
        reasoning: str,
        tags: List[str],
        id: str = None,
        strength: int = 1,
        is_active: bool = False,
        created_at: float = None,
        is_consolidated: bool = None,
        state_snapshot: dict = None,
        memory_scope: str = "public",
        useful_count: int = 0,
        useful_score: float = 0.0,
        last_recalled_at: float = 0.0,
        source_message_ids: List[str] | None = None,
        source_message_roles: List[str] | None = None,
        source_message_senders: List[dict] | None = None,
        source_message_is_bot: bool = False,
        primary_speaker_role: str = "",
        secondary_speaker_role: str = "",
        memory_perspective: str = "",
        source_start_ts: float = 0.0,
        source_end_ts: float = 0.0,
        event_start_ts: float = 0.0,
        event_end_ts: float = 0.0,
        event_time_confidence: str = TIME_CONFIDENCE_LOW,
    ):
        self.id = id or str(uuid.uuid4())
        self.memory_type = memory_type
        self.judgment = judgment
        self.reasoning = reasoning
        self.tags = tags if isinstance(tags, list) else []
        self.strength = strength
        self.is_active = is_active
        self.created_at = created_at or time.time()
        self.state_snapshot = state_snapshot or {}
        self.memory_scope = memory_scope or "public"
        self.useful_count = int(useful_count or 0)
        self.useful_score = float(useful_score or 0.0)
        self.last_recalled_at = float(last_recalled_at or 0.0)
        self.similarity = 0.0

        time_metadata = apply_time_metadata_defaults(
            {
                "source_message_ids": source_message_ids or [],
                "source_start_ts": source_start_ts,
                "source_end_ts": source_end_ts,
                "event_start_ts": event_start_ts,
                "event_end_ts": event_end_ts,
                "event_time_confidence": event_time_confidence,
            },
            created_at=self.created_at,
            text_for_inference="\n".join(
                [str(self.judgment or "").strip(), str(self.reasoning or "").strip()]
            ).strip(),
        )
        self.source_message_ids = normalize_source_message_ids(
            time_metadata.get("source_message_ids", [])
        )
        self.source_start_ts = float(time_metadata.get("source_start_ts", 0.0) or 0.0)
        self.source_end_ts = float(time_metadata.get("source_end_ts", 0.0) or 0.0)
        self.event_start_ts = float(time_metadata.get("event_start_ts", 0.0) or 0.0)
        self.event_end_ts = float(time_metadata.get("event_end_ts", 0.0) or 0.0)
        self.event_time_confidence = str(
            time_metadata.get("event_time_confidence", TIME_CONFIDENCE_LOW) or TIME_CONFIDENCE_LOW
        )
        self.source_message_roles = [
            str(role).strip()
            for role in self._parse_json_list(source_message_roles)
            if str(role or "").strip()
        ]
        self.source_message_senders = [
            sender
            for sender in self._parse_json_list(source_message_senders)
            if isinstance(sender, dict)
        ]
        self.source_message_is_bot = bool(source_message_is_bot)
        self.primary_speaker_role = str(primary_speaker_role or "").strip()
        self.secondary_speaker_role = str(secondary_speaker_role or "").strip()
        self.memory_perspective = str(memory_perspective or "").strip()

    def get_semantic_core(self) -> str:
        tags_text = KNOWLEDGE_CORE_SEPARATOR.join(self.tags)
        return f"{self.judgment}{KNOWLEDGE_CORE_SEPARATOR} {tags_text}"

    def to_dict(self) -> dict:
        tags_str = ", ".join(self.tags) if isinstance(self.tags, list) else ""
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "judgment": self.judgment,
            "reasoning": self.reasoning,
            "tags": tags_str,
            "strength": self.strength,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "useful_count": self.useful_count,
            "useful_score": self.useful_score,
            "last_recalled_at": self.last_recalled_at,
            "state_snapshot": json.dumps(self.state_snapshot) if self.state_snapshot else "{}",
            "memory_scope": self.memory_scope,
            "source_message_ids": json.dumps(self.source_message_ids, ensure_ascii=False),
            "source_message_roles": json.dumps(self.source_message_roles, ensure_ascii=False),
            "source_message_senders": json.dumps(self.source_message_senders, ensure_ascii=False),
            "source_message_is_bot": int(bool(self.source_message_is_bot)),
            "primary_speaker_role": self.primary_speaker_role,
            "secondary_speaker_role": self.secondary_speaker_role,
            "memory_perspective": self.memory_perspective,
            "source_start_ts": self.source_start_ts,
            "source_end_ts": self.source_end_ts,
            "event_start_ts": self.event_start_ts,
            "event_end_ts": self.event_end_ts,
            "event_time_confidence": self.event_time_confidence,
        }

    @staticmethod
    def _parse_json_dict(data) -> dict:
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                return {}
        if isinstance(data, dict):
            return data
        return {}

    @staticmethod
    def _parse_json_list(data) -> List[Any]:
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _parse_tags(tags_data) -> List[str]:
        if isinstance(tags_data, str):
            return [tag.strip() for tag in tags_data.split(",") if tag.strip()]
        if isinstance(tags_data, list):
            return tags_data
        return []

    @classmethod
    def from_dict(cls, data: dict) -> Optional["BaseMemory"]:
        if not isinstance(data, dict):
            raise ValidationError(f"输入必须是字典类型，实际为 {type(data)}")
        if "id" not in data:
            raise ValidationError("缺少必需字段: id")

        memory_type_str = data.get("memory_type")
        if not memory_type_str:
            return None
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            return None

        tags = cls._parse_tags(data.get("tags", []))
        judgment = data.get("judgment", data.get("content", data.get("emotion_type", "")))
        reasoning = data.get("reasoning", data.get("context", data.get("trigger_event", "")))

        try:
            return cls(
                memory_type=memory_type,
                judgment=judgment,
                reasoning=reasoning,
                tags=tags,
                id=data.get("id", str(uuid.uuid4())),
                strength=data.get("strength", 1),
                is_active=data.get("is_active", False),
                created_at=data.get("created_at", time.time()),
                state_snapshot=cls._parse_json_dict(data.get("state_snapshot", {})),
                memory_scope=data.get("memory_scope", "public"),
                useful_count=data.get("useful_count", 0),
                useful_score=data.get("useful_score", 0.0),
                last_recalled_at=data.get("last_recalled_at", 0.0),
                source_message_ids=data.get("source_message_ids", []),
                source_message_roles=cls._parse_json_list(data.get("source_message_roles", [])),
                source_message_senders=cls._parse_json_list(data.get("source_message_senders", [])),
                source_message_is_bot=bool(data.get("source_message_is_bot", False)),
                primary_speaker_role=data.get("primary_speaker_role", ""),
                secondary_speaker_role=data.get("secondary_speaker_role", ""),
                memory_perspective=data.get("memory_perspective", ""),
                source_start_ts=data.get("source_start_ts", 0.0),
                source_end_ts=data.get("source_end_ts", 0.0),
                event_start_ts=data.get("event_start_ts", 0.0),
                event_end_ts=data.get("event_end_ts", 0.0),
                event_time_confidence=data.get("event_time_confidence", TIME_CONFIDENCE_LOW),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValidationError(f"从字典创建记忆失败: {str(e)}")

    def __str__(self) -> str:
        return (
            "Memory("
            f"type={self.memory_type.value}, "
            f"id={self.id[:8]}..., "
            f"judgment='{self.judgment[:30]}...', "
            f"tags={self.tags})"
        )
