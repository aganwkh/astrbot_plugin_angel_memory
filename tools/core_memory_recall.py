import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent

from ..core.session_memory import MemoryItem
from ..core.utils.memory_formatter import MemoryFormatter
from ..core.utils.time_diagnostics import (
    analyze_recall_request,
    analyze_time_intent,
    build_time_filter_payload,
    get_event_diagnostic_store,
    preview_text,
    summarize_memory_records,
    tool_time_range_to_intent,
)
from ..llm_memory.models.data_models import BaseMemory

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def _memory_type_value(memory: BaseMemory) -> str:
    return (
        memory.memory_type.value
        if hasattr(memory.memory_type, "value")
        else str(memory.memory_type)
    )


@dataclass
class CoreMemoryRecallTool(FunctionTool):
    name: str = "core_memory_recall"
    description: str = (
        "当你需要主动回忆被保存的核心记忆、长期目标或关键事实时调用。"
        "当用户追问“昨天聊了什么”“昨晚还记得吗”“上周四那个事呢”“我原话是什么”时，"
        "必须优先调用这个工具，不要凭空猜测或直接回答“没有记录”。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "要返回的记忆数量",
                    "minimum": 1,
                },
                "query": {
                    "type": "string",
                    "description": "检索关键词，必填",
                    "minLength": 1,
                },
                "time_range": {
                    "type": "string",
                    "description": "可选。显式时间范围标准值。",
                    "enum": [
                        "",
                        "just_now",
                        "early_morning",
                        "today_morning",
                        "noon",
                        "today_afternoon",
                        "today_night",
                        "yesterday_early_morning",
                        "yesterday_morning",
                        "yesterday_noon",
                        "yesterday_afternoon",
                        "last_night",
                        "yesterday",
                        "a_few_days_ago",
                        "this_week",
                        "past_7_days",
                        "last_weekend",
                        "this_month",
                        "past_30_days",
                        "past_3_months",
                        "this_year",
                        "past_year",
                    ],
                },
                "original_user_text": {
                    "type": "string",
                    "description": "可选。用户原始问题文本，优先用于服务端时间解析。",
                },
                "original_query": {
                    "type": "string",
                    "description": "可选。未被压缩前的原始查询文本。",
                },
            },
            "required": ["limit", "query"],
        }
    )

    def __post_init__(self):
        self.logger = logger

    @staticmethod
    def _format_memories(memories: List[BaseMemory]) -> str:
        display_memories = [
            MemoryItem(
                id=str(getattr(mem, "id", "") or ""),
                memory_type=_memory_type_value(mem),
                judgment=str(getattr(mem, "judgment", "") or ""),
                reasoning=str(getattr(mem, "reasoning", "") or ""),
                tags=list(getattr(mem, "tags", []) or []),
                strength=int(getattr(mem, "strength", 0) or 0),
            )
            for mem in memories
        ]
        return MemoryFormatter.format_session_memories(display_memories)

    @staticmethod
    def _memory_debug_rows(memories: List[BaseMemory], inactive: bool = False) -> List[Dict[str, Any]]:
        rows = []
        for mem in memories[:10]:
            rows.append(
                {
                    "id": str(getattr(mem, "id", "") or ""),
                    "judgment_preview": preview_text(str(getattr(mem, "judgment", "") or ""), 80),
                    "is_active": bool(getattr(mem, "is_active", False)),
                    "strength": int(getattr(mem, "strength", 0) or 0),
                    "similarity": float(getattr(mem, "similarity", 0.0) or 0.0),
                    "created_at": float(getattr(mem, "created_at", 0.0) or 0.0),
                    "filter_reason": "inactive_filtered" if inactive else "retained",
                }
            )
        return rows

    @staticmethod
    def _sample_active_memories(memories: List[BaseMemory], limit: int) -> List[BaseMemory]:
        if not memories:
            return []

        population = list(memories)
        if limit >= len(population):
            return sorted(
                population,
                key=lambda mem: (
                    float(getattr(mem, "similarity", 0.0) or 0.0),
                    int(getattr(mem, "strength", 0) or 0),
                ),
                reverse=True,
            )

        weights = [max(1, int(getattr(mem, "strength", 0) or 0)) for mem in population]
        sampled_memories: List[BaseMemory] = []
        remaining_population = population.copy()
        remaining_weights = weights.copy()

        for _ in range(min(limit, len(remaining_population))):
            total_weight = sum(remaining_weights)
            if total_weight <= 0:
                sampled_memories.extend(random.sample(remaining_population, limit - len(sampled_memories)))
                break

            r = random.random() * total_weight
            cumulative = 0.0
            selected_idx = 0
            for idx, weight in enumerate(remaining_weights):
                cumulative += weight
                if r <= cumulative:
                    selected_idx = idx
                    break

            sampled_memories.append(remaining_population[selected_idx])
            del remaining_population[selected_idx]
            del remaining_weights[selected_idx]

        return sampled_memories

    def _resolve_time_intent(
        self,
        explicit_time_range: str,
        original_user_text: str,
        original_query: str,
        query: str,
    ) -> Tuple[str, Any]:
        candidates = [
            ("explicit_time_range", tool_time_range_to_intent(explicit_time_range)),
            ("original_user_text", analyze_time_intent(original_user_text)),
            ("original_query", analyze_time_intent(original_query)),
            ("tool_query", analyze_time_intent(query)),
        ]
        for source, intent in candidates:
            if intent.matched and build_time_filter_payload(intent).get("matched"):
                return source, intent
        return "none", tool_time_range_to_intent("", timezone_name="Asia/Shanghai")

    async def run(
        self,
        event: AstrMessageEvent,
        limit: int,
        query: str,
        time_range: str = "",
        original_user_text: str = "",
        original_query: str = "",
    ) -> str:
        if query is None or not str(query).strip():
            return "参数错误：query 为必填且不能为空。"

        if not hasattr(event, "plugin_context") or event.plugin_context is None:
            self.logger.error(f"{self.name}: 无法从事件中获取 plugin_context。")
            return "错误：内部服务错误，无法获取插件上下文。"

        plugin_context = event.plugin_context
        diagnostic_store = get_event_diagnostic_store(event)
        raw_user_input = str(
            original_user_text
            or diagnostic_store.get("raw_user_input", "")
            or getattr(getattr(event, "message_obj", None), "message_str", "")
            or getattr(event, "message_str", "")
            or ""
        ).strip()
        fallback_original_query = str(
            original_query
            or diagnostic_store.get("query_build", {}).get("intermediate_query", "")
            or query
        ).strip()
        tool_query = str(query).strip()

        try:
            memory_runtime = plugin_context.get_component("memory_runtime")
            if not memory_runtime:
                raise ValueError("memory_runtime 未在 PluginContext 中注册。")
            memory_scope = await plugin_context.resolve_memory_scope_from_event(event)
            if not isinstance(memory_scope, str):
                memory_scope = str(memory_scope)
        except Exception as e:
            self.logger.error(f"{self.name}: 无法获取上下文信息或 memory_runtime 实例: {e}")
            return "错误：无法确定当前会话 ID，主动回忆已拒绝。"

        explicit_time_intent = tool_time_range_to_intent(time_range)
        source_name, resolved_time_intent = self._resolve_time_intent(
            explicit_time_range=time_range,
            original_user_text=raw_user_input,
            original_query=fallback_original_query,
            query=tool_query,
        )
        resolved_time_filter = build_time_filter_payload(resolved_time_intent)
        recall_request = analyze_recall_request(raw_user_input)

        recall_tool_payload = {
            "raw_user_input": raw_user_input,
            "original_query": fallback_original_query,
            "tool_query": tool_query,
            "tool_time_range": str(time_range or ""),
            "time_intent_from_explicit": explicit_time_intent.to_dict(),
            "time_intent_from_raw_user_input": analyze_time_intent(raw_user_input).to_dict(),
            "time_intent_from_original_query": analyze_time_intent(fallback_original_query).to_dict(),
            "time_intent_from_tool_query": analyze_time_intent(tool_query).to_dict(),
            "resolved_time_intent_source": source_name,
            "resolved_time_filter": resolved_time_filter,
            "recall_request": recall_request,
            "query_build": diagnostic_store.get("query_build", {}),
            "query_pipeline": diagnostic_store.get("query_pipeline", {}),
        }

        try:
            candidate_limit = max(int(limit) * 3, 20)
            all_memories: List[BaseMemory] = await memory_runtime.comprehensive_recall(
                query=tool_query,
                fresh_limit=candidate_limit,
                event=event,
                memory_scope=memory_scope,
                time_filter=resolved_time_filter if resolved_time_filter.get("matched") else None,
            )

            active_memories = [mem for mem in all_memories if bool(getattr(mem, "is_active", False))]
            inactive_memories = [mem for mem in all_memories if not bool(getattr(mem, "is_active", False))]

            recall_tool_payload.update(
                {
                    "memory_scope": memory_scope,
                    "total_hits": len(all_memories),
                    "active_hits": len(active_memories),
                    "inactive_hits": len(inactive_memories),
                    "total_hit_summary": summarize_memory_records(all_memories),
                    "active_hit_summary": summarize_memory_records(active_memories),
                    "inactive_hit_summary": summarize_memory_records(inactive_memories),
                    "inactive_filtered_details": self._memory_debug_rows(inactive_memories, inactive=True),
                }
            )
            diagnostic_store["tool_recall"] = recall_tool_payload
            self.logger.info(
                f"[时间过滤诊断][core_memory_recall] payload="
                f"{json.dumps(recall_tool_payload, ensure_ascii=False)}"
            )

            if not all_memories:
                recall_tool_payload["final_returned_hits"] = 0
                recall_tool_payload["final_returned_mode"] = "empty"
                return "没有找到相关记忆。"

            if active_memories:
                sampled_memories = self._sample_active_memories(active_memories, int(limit))
                recall_tool_payload["final_returned_hits"] = len(sampled_memories)
                recall_tool_payload["final_returned_mode"] = "active_weighted"
                self.logger.info(
                    f"[时间过滤诊断][core_memory_recall返回] payload="
                    f"{json.dumps({'mode': 'active_weighted', 'returned': len(sampled_memories)}, ensure_ascii=False)}"
                )
                return self._format_memories(sampled_memories)

            fallback_memories = sorted(
                all_memories,
                key=lambda mem: (
                    float(getattr(mem, "similarity", 0.0) or 0.0),
                    int(getattr(mem, "strength", 0) or 0),
                ),
                reverse=True,
            )[: max(1, int(limit))]
            recall_tool_payload["final_returned_hits"] = len(fallback_memories)
            recall_tool_payload["final_returned_mode"] = "inactive_fallback"
            self.logger.warning(
                f"[时间过滤诊断][core_memory_recall降级返回] payload="
                f"{json.dumps({'mode': 'inactive_fallback', 'returned': len(fallback_memories)}, ensure_ascii=False)}"
            )
            return (
                "命中了相关记忆，但它们当前都不属于主动记忆；先返回最相关结果供参考：\n"
                + self._format_memories(fallback_memories)
            )

        except Exception as e:
            self.logger.error(f"{self.name}: 执行主动回忆失败: {e}", exc_info=True)
            return f"主动回忆失败：{str(e)}。请稍后再试。"
