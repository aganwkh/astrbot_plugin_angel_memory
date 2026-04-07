"""
检索词生成工具

统一处理笔记检索和记忆检索的查询词预处理：
1. 优先使用 angelheart 提供的 RAG 字段
2. 过滤助理名和别名
3. 从后往前保留 500 token
"""

import json
import re
from typing import List, Optional, Set, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .time_diagnostics import (
    analyze_time_intent,
    compare_time_intent,
    get_event_diagnostic_store,
    preview_text,
)


class QueryProcessor:
    """统一的检索词预处理工具类。"""

    def __init__(self):
        self.logger = logger

    def extract_rag_fields(self, event: AstrMessageEvent) -> dict:
        """从 angelheart_context 中提取 RAG 字段。"""
        rag_fields = {"entities": [], "facts": [], "keywords": []}

        try:
            if hasattr(event, "angelheart_context") and event.angelheart_context:
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})

                for field in rag_fields.keys():
                    field_value = secretary_decision.get(field, [])
                    if isinstance(field_value, list):
                        rag_fields[field] = [
                            str(item).strip()
                            for item in field_value
                            if item and str(item).strip()
                        ]
                    elif isinstance(field_value, str) and field_value.strip():
                        rag_fields[field] = [field_value.strip()]

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.debug(f"无法提取 RAG 字段: {e}")

        return rag_fields

    def _extract_assistant_names(self, event: AstrMessageEvent) -> Set[str]:
        """提取助理的人格名和别名。"""
        names = set()

        try:
            if hasattr(event, "angelheart_context"):
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})

                persona_name = secretary_decision.get("persona_name", "").strip()
                if persona_name:
                    names.add(persona_name)

                alias = secretary_decision.get("alias", "")
                if isinstance(alias, str) and alias.strip():
                    if "|" in alias:
                        for name in alias.split("|"):
                            name = name.strip()
                            if name:
                                names.add(name)
                    else:
                        names.add(alias.strip())
                elif isinstance(alias, list):
                    for item in alias:
                        if isinstance(item, str) and item.strip():
                            names.add(item.strip())

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.debug(f"无法提取助理名称: {e}")

        return names

    def _filter_assistant_names(self, query: str, names: Set[str]) -> str:
        """过滤查询中出现的助理名。"""
        if not names:
            return query

        filtered_query = query
        for name in names:
            pattern = re.escape(name)
            filtered_query = re.sub(pattern, "", filtered_query, flags=re.IGNORECASE)

        return filtered_query

    def _truncate_text(self, text: str, max_tokens: int = 500) -> str:
        """从后向前保留指定 token 数量。"""
        if not text.strip():
            return ""

        try:
            from ...llm_memory.utils.token_utils import truncate_by_tokens_from_end

            return truncate_by_tokens_from_end(text, max_tokens)
        except Exception as e:
            self.logger.warning(f"Token 截断处理失败: {e}")
            if len(text) <= max_tokens * 4:
                return text
            return text[-(max_tokens * 4) :]

    def _clean_text(self, text: str) -> str:
        """清理多余空白。"""
        if not text:
            return text
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _build_rag_query(self, rag_fields: dict) -> str:
        """按 entities > facts > keywords 拼接 RAG 查询。"""
        query_parts = []
        for field_name in ["entities", "facts", "keywords"]:
            field_values = rag_fields.get(field_name, [])
            if field_values:
                query_parts.extend(
                    [str(value) for value in field_values if value and str(value).strip()]
                )
        return " ".join(query_parts).strip()

    def _store_query_diagnostic(
        self,
        event: Optional[AstrMessageEvent],
        query_kind: str,
        payload: dict,
    ) -> None:
        if event is None:
            return
        diagnostic_store = get_event_diagnostic_store(event)
        query_pipeline = diagnostic_store.setdefault("query_pipeline", {})
        query_pipeline[str(query_kind or "generic")] = payload

    def process_query(
        self, query: str, event: AstrMessageEvent, query_kind: str = "generic"
    ) -> str:
        """统一处理检索查询。"""
        if not query or not query.strip():
            return query

        original_query = query

        try:
            diagnostic_store = get_event_diagnostic_store(event)
            raw_user_input = str(diagnostic_store.get("raw_user_input", "") or "")
            rag_fields = self.extract_rag_fields(event)
            rag_query = self._build_rag_query(rag_fields)

            final_query = rag_query if rag_query else original_query

            preprocess_threshold_characters = 100
            if rag_query and len(final_query.strip()) <= preprocess_threshold_characters:
                processor_diagnostic = {
                    "query_kind": query_kind,
                    "original_query": original_query,
                    "original_query_preview": preview_text(original_query, 160),
                    "rag_fields": rag_fields,
                    "rag_query": rag_query,
                    "rag_query_preview": preview_text(rag_query, 160),
                    "final_query": final_query,
                    "final_query_preview": preview_text(final_query, 160),
                    "used_rag_query": True,
                    "raw_to_original": compare_time_intent(raw_user_input, original_query),
                    "original_to_final": compare_time_intent(original_query, final_query),
                    "raw_intent": analyze_time_intent(raw_user_input).to_dict(),
                    "original_intent": analyze_time_intent(original_query).to_dict(),
                    "final_intent": analyze_time_intent(final_query).to_dict(),
                    "preprocess_skipped": True,
                }
                self._store_query_diagnostic(event, query_kind, processor_diagnostic)
                self.logger.info(
                    f"[时间过滤诊断][query处理] kind={query_kind} payload="
                    f"{json.dumps(processor_diagnostic, ensure_ascii=False)}"
                )
                return final_query

            assistant_names = self._extract_assistant_names(event)
            if assistant_names:
                final_query = self._filter_assistant_names(final_query, assistant_names)
                final_query = self._clean_text(final_query)

            if final_query.strip():
                final_query = self._truncate_text(final_query, 500)
                final_query = self._clean_text(final_query)

            processor_diagnostic = {
                "query_kind": query_kind,
                "original_query": original_query,
                "original_query_preview": preview_text(original_query, 160),
                "rag_fields": rag_fields,
                "rag_query": rag_query,
                "rag_query_preview": preview_text(rag_query, 160),
                "final_query": final_query,
                "final_query_preview": preview_text(final_query, 160),
                "used_rag_query": bool(rag_query),
                "raw_to_original": compare_time_intent(raw_user_input, original_query),
                "original_to_final": compare_time_intent(original_query, final_query),
                "raw_intent": analyze_time_intent(raw_user_input).to_dict(),
                "original_intent": analyze_time_intent(original_query).to_dict(),
                "final_intent": analyze_time_intent(final_query).to_dict(),
                "preprocess_skipped": False,
            }
            self._store_query_diagnostic(event, query_kind, processor_diagnostic)
            self.logger.info(
                f"[时间过滤诊断][query处理] kind={query_kind} payload="
                f"{json.dumps(processor_diagnostic, ensure_ascii=False)}"
            )
            return final_query

        except Exception as e:
            processor_diagnostic = {
                "query_kind": query_kind,
                "original_query": original_query,
                "original_query_preview": preview_text(original_query, 160),
                "final_query": original_query,
                "final_query_preview": preview_text(original_query, 160),
                "error": str(e),
                "raw_intent": analyze_time_intent(raw_user_input if "raw_user_input" in locals() else "").to_dict(),
                "original_intent": analyze_time_intent(original_query).to_dict(),
                "final_intent": analyze_time_intent(original_query).to_dict(),
            }
            self._store_query_diagnostic(event, f"{query_kind}_error", processor_diagnostic)
            self.logger.error(
                f"[时间过滤诊断][query处理异常] kind={query_kind} payload="
                f"{json.dumps(processor_diagnostic, ensure_ascii=False)}"
            )
            self.logger.error(f"查询词预处理失败: {e}")
            return original_query

    async def _precompute_rag_vector(
        self, rag_query: str, event: AstrMessageEvent
    ) -> Optional[List[float]]:
        """为处理后的查询预计算向量。"""
        if not rag_query.strip():
            return None

        from ..plugin_context import PluginContext

        plugin_context: Optional[PluginContext] = getattr(event, "plugin_context", None)
        if plugin_context is None:
            self.logger.debug("无法从事件中获取 plugin_context，跳过 RAG 向量预计算")
            return None

        vector_store = plugin_context.get_vector_store()
        if vector_store is None:
            self.logger.debug("plugin_context 中未找到有效的 vector_store，跳过 RAG 向量预计算")
            return None

        try:
            return await vector_store.embed_single_document(rag_query, is_query=True)
        except Exception as e:
            self.logger.debug(f"RAG 向量预计算失败: {e}")
            return None

    def process_query_for_memory(self, query: str, event: AstrMessageEvent) -> str:
        return self.process_query(query, event, query_kind="memory")

    def process_query_for_notes(self, query: str, event: AstrMessageEvent) -> str:
        return self.process_query(query, event, query_kind="note")

    async def process_query_for_memory_with_vector(
        self, query: str, event: AstrMessageEvent
    ) -> Tuple[str, Optional[List[float]]]:
        processed_query = self.process_query_for_memory(query, event)
        vector = await self._precompute_rag_vector(processed_query, event)
        return processed_query, vector

    async def process_query_for_notes_with_vector(
        self, query: str, event: AstrMessageEvent
    ) -> Tuple[str, Optional[List[float]]]:
        processed_query = self.process_query_for_notes(query, event)
        vector = await self._precompute_rag_vector(processed_query, event)
        return processed_query, vector


_query_processor_instance = None


def get_query_processor() -> QueryProcessor:
    """获取 QueryProcessor 全局单例。"""
    global _query_processor_instance
    if _query_processor_instance is None:
        _query_processor_instance = QueryProcessor()
    return _query_processor_instance
