"""
记忆格式化器

负责将记忆格式化为统一的文本形式，便于阅读和理解。
"""

from typing import List, Dict, Any, Optional

from ..session_memory import MemoryItem
from ..config import MemoryConstants


class MemoryFormatter:
    """记忆格式化器"""

    MEMORY_TYPE_NAMES = MemoryConstants.MEMORY_TYPE_NAMES

    @staticmethod
    def _memory_perspective_label(memory: MemoryItem) -> str:
        perspective = str(getattr(memory, "memory_perspective", "") or "").strip()
        label_map = {
            "user_fact": "[用户事实]",
            "assistant_fact": "[助理事实]",
            "shared_dialogue": "[双方对话]",
            "user_said": "[用户说]",
            "assistant_said": "[助理说]",
        }
        label = label_map.get(perspective, "")
        if label:
            return label

        source_roles = [
            str(role).strip()
            for role in (getattr(memory, "source_message_roles", []) or [])
            if str(role or "").strip()
        ]
        source_roles_set = set(source_roles)
        primary_role = str(getattr(memory, "primary_speaker_role", "") or "").strip()
        secondary_role = str(getattr(memory, "secondary_speaker_role", "") or "").strip()

        if "assistant" in source_roles_set and "user" in source_roles_set:
            return "[双方对话]"
        if primary_role == "assistant" or (
            not source_roles_set and bool(getattr(memory, "source_message_is_bot", False))
        ):
            return "[助理说]"
        if primary_role == "user" or "user" in source_roles_set:
            return "[用户说]"
        if secondary_role == "assistant":
            return "[助理说]"
        if secondary_role == "user":
            return "[用户说]"
        return ""

    @staticmethod
    def _format_memory_body(memory: MemoryItem) -> str:
        judgment = str(getattr(memory, "judgment", "") or "").strip()
        reasoning = str(getattr(memory, "reasoning", "") or "").strip()
        prefix = MemoryFormatter._memory_perspective_label(memory)
        body = f"{prefix}{judgment}" if prefix else judgment
        if reasoning:
            body = f"{body}\n——因为{reasoning}"
        return body

    @staticmethod
    def format_single_memory(memory: MemoryItem) -> str:
        """格式化单条记忆。"""
        return MemoryFormatter._format_memory_body(memory)

    @staticmethod
    def format_memories_by_type(memories: List[MemoryItem]) -> Dict[str, List[str]]:
        """按类型分组格式化记忆。"""
        grouped_memories: Dict[str, List[str]] = {}
        for memory in memories:
            type_value = memory.memory_type
            type_name = MemoryFormatter.MEMORY_TYPE_NAMES.get(type_value, type_value)
            grouped_memories.setdefault(type_name, []).append(
                MemoryFormatter._format_memory_body(memory)
            )
        return grouped_memories

    @staticmethod
    def format_memories_for_prompt(
        memories: List[MemoryItem],
        useful_memory_ids: Optional[List[str]] = None,
        new_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """格式化记忆用于提示词。"""
        if not memories and not new_memories:
            return ""

        memory_map = {memory.id: memory for memory in memories}
        memories_to_format: List[MemoryItem] = []

        if useful_memory_ids:
            memories_to_format.extend(
                memory_map[mid] for mid in useful_memory_ids if mid in memory_map
            )

        if new_memories:
            memories_to_format.extend(
                MemoryItem(
                    id=nm.get("id", "new"),
                    memory_type=nm.get("type", "knowledge"),
                    judgment=nm.get("judgment", ""),
                    reasoning=nm.get("reasoning", ""),
                    tags=nm.get("tags", []),
                    strength=nm.get("strength", 0.5),
                    source_message_ids=nm.get("source_message_ids", []),
                    source_message_roles=nm.get("source_message_roles", []),
                    source_message_senders=nm.get("source_message_senders", []),
                    source_message_is_bot=bool(nm.get("source_message_is_bot", False)),
                    primary_speaker_role=nm.get("primary_speaker_role", ""),
                    secondary_speaker_role=nm.get("secondary_speaker_role", ""),
                    memory_perspective=nm.get("memory_perspective", ""),
                )
                for nm in new_memories
            )

        if not memories_to_format:
            return ""

        grouped_memories = MemoryFormatter.format_memories_by_type(memories_to_format)
        formatted_lines = ["[相关记忆]"]
        for memory_type, formatted_memories in grouped_memories.items():
            formatted_lines.append(f"\n[{memory_type}]")
            formatted_lines.extend(f"\n{text}" for text in formatted_memories)

        return "".join(formatted_lines)

    @staticmethod
    def _deduplicate_memories(memories: List[MemoryItem]) -> List[MemoryItem]:
        """按 judgment 去重。"""
        seen_judgments = set()
        deduplicated: List[MemoryItem] = []
        for memory in memories:
            normalized_judgment = str(memory.judgment or "").strip().lower()
            if normalized_judgment not in seen_judgments:
                seen_judgments.add(normalized_judgment)
                deduplicated.append(memory)
        return deduplicated

    @staticmethod
    def format_session_memories(memories: List[MemoryItem]) -> str:
        """格式化会话短期记忆。"""
        if not memories:
            return ""

        deduplicated_memories = MemoryFormatter._deduplicate_memories(memories)
        grouped_memories = MemoryFormatter.format_memories_by_type(deduplicated_memories)

        formatted_lines = ["[相关记忆]"]
        for memory_type, formatted_memories in grouped_memories.items():
            formatted_lines.append(f"\n[{memory_type}]")
            formatted_lines.extend(f"\n{text}" for text in formatted_memories)

        return "".join(formatted_lines)

    @staticmethod
    def format_memories_for_display(memories: List[MemoryItem]) -> str:
        """格式化记忆用于显示。"""
        if not memories:
            return "暂无记忆"

        from .memory_id_resolver import MemoryIDResolver

        grouped: Dict[str, List[MemoryItem]] = {}
        for memory in memories:
            type_value = memory.memory_type
            type_name = MemoryFormatter.MEMORY_TYPE_NAMES.get(type_value, type_value)
            grouped.setdefault(type_name, []).append(memory)

        display_lines: List[str] = []
        for memory_type, memory_list in grouped.items():
            display_lines.append(f"\n=== {memory_type} ===")
            for i, memory in enumerate(memory_list, 1):
                short_id = MemoryIDResolver.generate_short_id(memory.id)
                display_lines.append(
                    f"\n{i}. [id:{short_id}]{MemoryFormatter._format_memory_body(memory)}"
                )

        return "".join(display_lines)