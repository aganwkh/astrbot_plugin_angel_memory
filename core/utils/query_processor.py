"""
查询词处理工具。

统一处理记忆检索和笔记检索的查询词预处理：
1. 先兼容 angelheart_context 的 dict / str 形态，并解析 secretary 字段
2. 再决定使用 RAG 查询还是原始查询
3. 最后统一执行助理名过滤、清理与截断
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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

    @staticmethod
    def _normalize_string_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item or "").strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _load_angelheart_context(
        self,
        event: AstrMessageEvent,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raw_context = getattr(event, "angelheart_context", None)
        context_type = type(raw_context).__name__
        diagnostic = {
            "exists": raw_context is not None,
            "angelheart_context_type": context_type,
            "parse_mode": "",
            "parse_error": "",
            "parsed_type": "",
        }

        if raw_context is None:
            diagnostic["parse_mode"] = "missing"
            return {}, diagnostic

        if isinstance(raw_context, dict):
            diagnostic["parse_mode"] = "dict"
            diagnostic["parsed_type"] = "dict"
            return raw_context, diagnostic

        if isinstance(raw_context, str):
            if not raw_context.strip():
                diagnostic["parse_mode"] = "empty_str"
                return {}, diagnostic
            try:
                parsed = json.loads(raw_context)
            except (json.JSONDecodeError, TypeError) as e:
                diagnostic["parse_mode"] = "json_error"
                diagnostic["parse_error"] = str(e)
                return {}, diagnostic
            diagnostic["parse_mode"] = "json_str"
            diagnostic["parsed_type"] = type(parsed).__name__
            if isinstance(parsed, dict):
                return parsed, diagnostic
            diagnostic["parse_mode"] = "non_dict_json"
            return {}, diagnostic

        diagnostic["parse_mode"] = "unsupported"
        return {}, diagnostic

    def _build_secretary_diagnostic(
        self,
        context_payload: Dict[str, Any],
        context_diagnostic: Dict[str, Any],
    ) -> Dict[str, Any]:
        secretary_decision = context_payload.get("secretary_decision", {})
        parse_failed = context_diagnostic.get("parse_mode") in {
            "json_error",
            "unsupported",
            "non_dict_json",
        }
        secretary_exists = isinstance(secretary_decision, dict) and bool(secretary_decision)
        if not isinstance(secretary_decision, dict):
            secretary_decision = {}

        field_status: Dict[str, str] = {}
        field_values: Dict[str, Any] = {}
        for field_name in ["topic", "entities", "facts", "keywords", "persona_name", "alias"]:
            raw_value = secretary_decision.get(field_name)
            field_values[field_name] = raw_value
            if parse_failed:
                field_status[field_name] = "parse_failed"
            elif not context_diagnostic.get("exists"):
                field_status[field_name] = "missing_context"
            elif not secretary_exists:
                field_status[field_name] = "missing_secretary_decision"
            elif raw_value is None:
                field_status[field_name] = "missing_field"
            elif isinstance(raw_value, str) and not raw_value.strip():
                field_status[field_name] = "upstream_empty"
            elif isinstance(raw_value, list) and not any(
                str(item or "").strip() for item in raw_value
            ):
                field_status[field_name] = "upstream_empty"
            else:
                field_status[field_name] = "parsed"

        return {
            **context_diagnostic,
            "secretary_decision_exists": secretary_exists,
            "secretary_field_status": field_status,
            "secretary_field_values": field_values,
        }

    def _log_secretary_diagnostic(
        self,
        event: AstrMessageEvent,
        secretary_diagnostic: Dict[str, Any],
    ) -> None:
        diagnostic_store = get_event_diagnostic_store(event)
        query_pipeline = diagnostic_store.setdefault("query_pipeline", {})
        query_pipeline["angelheart_context"] = secretary_diagnostic

        log_prefix = "[时间过滤诊断][QueryProcessor上下文解析] payload="
        payload = json.dumps(secretary_diagnostic, ensure_ascii=False)
        if secretary_diagnostic.get("parse_error"):
            self.logger.warning(f"{log_prefix}{payload}")
        else:
            self.logger.info(f"{log_prefix}{payload}")

    def _extract_rag_fields_from_context(
        self,
        context_payload: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        rag_fields = {"entities": [], "facts": [], "keywords": []}
        secretary_decision = context_payload.get("secretary_decision", {})
        if not isinstance(secretary_decision, dict):
            return rag_fields

        for field_name in rag_fields.keys():
            rag_fields[field_name] = self._normalize_string_list(
                secretary_decision.get(field_name)
            )
        return rag_fields

    def _extract_assistant_names_from_context(
        self,
        context_payload: Dict[str, Any],
    ) -> Set[str]:
        names = set()
        secretary_decision = context_payload.get("secretary_decision", {})
        if not isinstance(secretary_decision, dict):
            return names

        persona_name = str(secretary_decision.get("persona_name", "") or "").strip()
        if persona_name:
            names.add(persona_name)

        alias = secretary_decision.get("alias", "")
        if isinstance(alias, str) and alias.strip():
            if "|" in alias:
                for name in alias.split("|"):
                    normalized = name.strip()
                    if normalized:
                        names.add(normalized)
            else:
                names.add(alias.strip())
        elif isinstance(alias, list):
            for item in alias:
                normalized = str(item or "").strip()
                if normalized:
                    names.add(normalized)

        return names

    def _prepare_secretary_context(
        self,
        event: AstrMessageEvent,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[str]], Set[str]]:
        context_payload, context_diagnostic = self._load_angelheart_context(event)
        secretary_diagnostic = self._build_secretary_diagnostic(
            context_payload,
            context_diagnostic,
        )
        rag_fields = self._extract_rag_fields_from_context(context_payload)
        secretary_diagnostic["rag_fields"] = rag_fields
        self._log_secretary_diagnostic(event, secretary_diagnostic)
        assistant_names = self._extract_assistant_names_from_context(context_payload)
        return context_payload, secretary_diagnostic, rag_fields, assistant_names

    def extract_rag_fields(self, event: AstrMessageEvent) -> dict:
        """从 angelheart_context 中提取 RAG 字段。"""
        _context_payload, _secretary_diagnostic, rag_fields, _assistant_names = (
            self._prepare_secretary_context(event)
        )
        return rag_fields

    def _extract_assistant_names(self, event: AstrMessageEvent) -> Set[str]:
        """提取助理的人格名和别名。"""
        context_payload, _context_diagnostic = self._load_angelheart_context(event)
        return self._extract_assistant_names_from_context(context_payload)

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
        self,
        query: str,
        event: AstrMessageEvent,
        query_kind: str = "generic",
    ) -> str:
        """统一处理检索查询。"""
        if not query or not query.strip():
            return query

        original_query = query

        try:
            diagnostic_store = get_event_diagnostic_store(event)
            raw_user_input = str(diagnostic_store.get("raw_user_input", "") or "")
            (
                _context_payload,
                secretary_diagnostic,
                rag_fields,
                assistant_names,
            ) = self._prepare_secretary_context(event)
            rag_query = self._build_rag_query(rag_fields)

            selected_query = rag_query if rag_query else original_query
            preprocess_threshold_characters = 100
            preprocess_skipped = bool(
                rag_query and len(selected_query.strip()) <= preprocess_threshold_characters
            )

            assistant_filtered_query = selected_query
            if assistant_names:
                assistant_filtered_query = self._filter_assistant_names(
                    assistant_filtered_query,
                    assistant_names,
                )

            final_query = self._clean_text(assistant_filtered_query)
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
                "selected_query": selected_query,
                "selected_query_preview": preview_text(selected_query, 160),
                "final_query": final_query,
                "final_query_preview": preview_text(final_query, 160),
                "used_rag_query": bool(rag_query),
                "assistant_names": sorted(assistant_names),
                "assistant_filter_applied": bool(assistant_names),
                "preprocess_skipped": preprocess_skipped,
                "context_parse_mode": secretary_diagnostic.get("parse_mode", ""),
                "secretary_decision_exists": secretary_diagnostic.get(
                    "secretary_decision_exists",
                    False,
                ),
                "raw_to_original": compare_time_intent(raw_user_input, original_query),
                "original_to_selected": compare_time_intent(original_query, selected_query),
                "selected_to_final": compare_time_intent(selected_query, final_query),
                "raw_intent": analyze_time_intent(raw_user_input).to_dict(),
                "original_intent": analyze_time_intent(original_query).to_dict(),
                "selected_intent": analyze_time_intent(selected_query).to_dict(),
                "final_intent": analyze_time_intent(final_query).to_dict(),
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
                "raw_intent": analyze_time_intent(
                    raw_user_input if "raw_user_input" in locals() else ""
                ).to_dict(),
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
        self,
        rag_query: str,
        event: AstrMessageEvent,
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
        self,
        query: str,
        event: AstrMessageEvent,
    ) -> Tuple[str, Optional[List[float]]]:
        processed_query = self.process_query_for_memory(query, event)
        vector = await self._precompute_rag_vector(processed_query, event)
        return processed_query, vector

    async def process_query_for_notes_with_vector(
        self,
        query: str,
        event: AstrMessageEvent,
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
