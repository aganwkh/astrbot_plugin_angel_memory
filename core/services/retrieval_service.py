import json
from typing import Any, Dict, Optional

from astrbot.api.event import AstrMessageEvent

from ..utils.memory_id_resolver import MemoryIDResolver
from ..utils.time_diagnostics import (
    get_event_diagnostic_store,
    preview_text,
    summarize_memory_records,
    summarize_note_records,
)


class DeepMindRetrievalService:
    """DeepMind 的检索相关职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind

    def parse_memory_context(
        self, event: AstrMessageEvent
    ) -> Optional[Dict[str, Any]]:
        if not hasattr(event, "angelmemory_context"):
            return None

        if event.angelmemory_context is None:
            return None

        try:
            context_data = json.loads(event.angelmemory_context)
            return {
                "session_id": context_data["session_id"],
                "query": context_data.get("recall_query", ""),
                "user_list": context_data.get("user_list", []),
                "raw_chat_records": context_data.get("raw_chat_records", []),
                "raw_memories": context_data.get("raw_memories", []),
                "raw_notes": context_data.get("raw_notes", []),
                "core_topic": context_data.get("core_topic", ""),
                "memory_id_mapping": context_data.get("memory_id_mapping", {}),
                "note_id_mapping": context_data.get("note_id_mapping", {}),
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.deepmind.logger.warning(f"解析记忆上下文失败: {e}")
            return None

    async def retrieve_memories_and_notes(
        self,
        event: AstrMessageEvent,
        query: str,
        precompute_vectors: bool = False,
        recall_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        deepmind = self.deepmind
        diagnostic_store = get_event_diagnostic_store(event)
        retrieval_diagnostic = diagnostic_store.setdefault("retrieval", {})
        time_intent = diagnostic_store.get("time_intent", {})
        recall_policy = recall_policy or getattr(event, "_angel_memory_recall_policy", {}) or {}
        recall_request = diagnostic_store.get("recall_request", {})
        time_filter = recall_policy.get("time_filter", {}) if isinstance(recall_policy, dict) else {}
        strict_time_recall = bool(recall_policy.get("strict_time_recall")) if isinstance(recall_policy, dict) else False
        prefer_raw_chat_only = bool(recall_policy.get("prefer_raw_chat_only")) if isinstance(recall_policy, dict) else False
        skip_notes = bool(recall_policy.get("skip_notes")) if isinstance(recall_policy, dict) else False
        requested_note_time_filter = time_filter if strict_time_recall else {}
        note_time_filter_capability = {
            "supported": False,
            "requested": bool(requested_note_time_filter),
            "requested_time_filter": requested_note_time_filter,
            "applied": False,
            "note": "note 检索链当前不支持 time_filter，不能把 note 结果视为时间过滤后的结果。",
        }
        retrieval_diagnostic["note_time_filter_capability"] = note_time_filter_capability

        if precompute_vectors:
            memory_query, memory_vector = (
                await deepmind.query_processor.process_query_for_memory_with_vector(
                    query, event
                )
            )
        else:
            memory_query = deepmind.query_processor.process_query_for_memory(query, event)
            memory_vector = None

        long_term_memories = []
        if prefer_raw_chat_only:
            memory_call_payload = {
                "memory_query": memory_query,
                "memory_query_preview": preview_text(memory_query, 160),
                "memory_scope": "",
                "per_type_limit": 0,
                "final_limit": 0,
                "has_precomputed_vector": memory_vector is not None,
                "time_intent": time_intent,
                "recall_request": recall_request,
                "time_filter_passed": bool(time_filter.get("matched")),
                "time_filter": time_filter,
                "skipped": True,
                "skip_reason": "原始聊天回顾优先，跳过自动长期记忆召回。",
            }
            retrieval_diagnostic["memory_call"] = memory_call_payload
            retrieval_diagnostic["memory_result"] = summarize_memory_records([])
            deepmind.logger.info(
                "[时间过滤诊断][长期记忆检索入参] payload="
                f"{json.dumps(memory_call_payload, ensure_ascii=False)}"
            )
        elif deepmind.memory_system:
            try:
                memory_scope = await deepmind.plugin_context.resolve_memory_scope_from_event(
                    event
                )
                rag_fields = deepmind.query_processor.extract_rag_fields(event)
                entities = rag_fields.get("entities", [])

                dynamic_limit = deepmind.CHAINED_RECALL_PER_TYPE_LIMIT
                if deepmind.soul:
                    try:
                        dynamic_limit = deepmind.soul.get_value("RecallDepth")
                        deepmind.logger.info(
                            f"🧠 灵魂回忆深度: {dynamic_limit} "
                            f"(E={deepmind.soul.energy['RecallDepth']:.1f})"
                        )
                    except Exception as e:
                        deepmind.logger.warning(f"获取灵魂参数失败，使用默认值: {e}")

                memory_call_payload = {
                    "memory_query": memory_query,
                    "memory_query_preview": preview_text(memory_query, 160),
                    "entities": list(entities or [])[:10],
                    "memory_scope": str(memory_scope or ""),
                    "per_type_limit": int(dynamic_limit),
                    "final_limit": int(dynamic_limit * 1.5),
                    "has_precomputed_vector": memory_vector is not None,
                    "time_intent": time_intent,
                    "recall_request": recall_request,
                    "time_filter_passed": bool(time_filter.get("matched") and strict_time_recall),
                    "time_filter": time_filter if strict_time_recall else {},
                    "time_filter_note": "命中时间回顾问题时，将时间窗透传到底层检索函数。",
                }
                retrieval_diagnostic["memory_call"] = memory_call_payload
                deepmind.logger.info(
                    "[时间过滤诊断][长期记忆检索入参] payload="
                    f"{json.dumps(memory_call_payload, ensure_ascii=False)}"
                )

                long_term_memories = await deepmind.memory_system.chained_recall(
                    query=memory_query,
                    entities=entities,
                    per_type_limit=int(dynamic_limit),
                    final_limit=int(dynamic_limit * 1.5),
                    event=event,
                    vector=memory_vector,
                    memory_scope=memory_scope,
                    time_filter=time_filter if strict_time_recall else None,
                )

                memory_result_payload = summarize_memory_records(long_term_memories)
                retrieval_diagnostic["memory_result"] = memory_result_payload
                deepmind.logger.info(
                    "[时间过滤诊断][长期记忆检索结果] payload="
                    f"{json.dumps(memory_result_payload, ensure_ascii=False)}"
                )

                if deepmind.soul:
                    snapshots = [
                        mem.state_snapshot
                        for mem in long_term_memories
                        if hasattr(mem, "state_snapshot") and mem.state_snapshot
                    ]
                    if snapshots:
                        deepmind.soul.resonate(snapshots)

            except Exception as e:
                retrieval_diagnostic["memory_error"] = {"error": str(e)}
                deepmind.logger.error(f"链式召回失败，跳过记忆检索: {e}")
                long_term_memories = []

        secretary_decision = {}
        try:
            if hasattr(event, "angelheart_context") and event.angelheart_context is not None:
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.deepmind.logger.warning(f"无法获取 secretary_decision 信息: {e}")

        if precompute_vectors:
            note_query, note_vector = (
                await deepmind.query_processor.process_query_for_notes_with_vector(
                    query, event
                )
            )
        else:
            note_query = deepmind.query_processor.process_query_for_notes(query, event)
            note_vector = None

        candidate_notes = []
        if skip_notes:
            note_call_payload = {
                "note_query": note_query,
                "note_query_preview": preview_text(note_query, 160),
                "recall_count": 0,
                "top_k": 0,
                "has_precomputed_vector": note_vector is not None,
                "time_intent": time_intent,
                "recall_request": recall_request,
                "time_filter_passed": bool(time_filter.get("matched") and strict_time_recall),
                "time_filter": time_filter if strict_time_recall else {},
                "time_filter_supported": False,
                "time_filter_requested": bool(requested_note_time_filter),
                "time_filter_requested_but_unsupported": bool(requested_note_time_filter),
                "time_filter_status_note": "note 检索链当前不支持 time_filter；即使本轮存在时间窗，note 结果也未按时间过滤。",
                "time_filter_note": "note 检索链当前不支持 time_filter；本轮因时间回顾优先策略直接跳过 note 检索。",
                "skipped": True,
                "skip_reason": "时间窗回顾问题优先避免笔记混入。",
            }
            retrieval_diagnostic["note_call"] = note_call_payload
            retrieval_diagnostic["note_result"] = {
                "summary": summarize_note_records([]),
                "time_filter_supported": False,
                "time_filter_requested": bool(requested_note_time_filter),
                "time_filter_requested_but_unsupported": bool(requested_note_time_filter),
                "time_filter_applied": False,
                "note": "note 检索链当前不支持 time_filter，且本轮已被跳过。",
            }
            deepmind.logger.info(
                "[时间过滤诊断][笔记检索入参] payload="
                f"{json.dumps(note_call_payload, ensure_ascii=False)}"
            )
        elif deepmind.note_service:
            note_call_payload = {
                "note_query": note_query,
                "note_query_preview": preview_text(note_query, 160),
                "recall_count": int(deepmind.note_candidate_top_k),
                "top_k": int(deepmind.note_candidate_top_k),
                "has_precomputed_vector": note_vector is not None,
                "time_intent": time_intent,
                "recall_request": recall_request,
                "time_filter_passed": False,
                "time_filter": requested_note_time_filter,
                "time_filter_supported": False,
                "time_filter_requested": bool(requested_note_time_filter),
                "time_filter_requested_but_unsupported": bool(requested_note_time_filter),
                "time_filter_note": "时间回顾问题默认跳过笔记；普通问题保持原有笔记检索。",
            }
            retrieval_diagnostic["note_call"] = note_call_payload
            deepmind.logger.info(
                "[时间过滤诊断][笔记检索入参] payload="
                f"{json.dumps(note_call_payload, ensure_ascii=False)}"
            )

            candidate_notes = await deepmind.note_service.search_notes_by_top_k(
                query=note_query,
                recall_count=deepmind.note_candidate_top_k,
                top_k=deepmind.note_candidate_top_k,
                vector=note_vector,
                time_filter=requested_note_time_filter or None,
            )

            note_result_payload = {
                "summary": summarize_note_records(candidate_notes),
                "time_filter_supported": False,
                "time_filter_requested": bool(requested_note_time_filter),
                "time_filter_requested_but_unsupported": bool(requested_note_time_filter),
                "time_filter_applied": False,
                "note": "note 检索结果未应用 time_filter，请勿把它视为按时间窗过滤后的候选。",
            }
            retrieval_diagnostic["note_result"] = note_result_payload
            deepmind.logger.info(
                "[时间过滤诊断][笔记检索结果] payload="
                f"{json.dumps(note_result_payload, ensure_ascii=False)}"
            )

        memory_id_mapping = {}
        if long_term_memories:
            memory_id_mapping = MemoryIDResolver.generate_id_mapping(
                [memory.to_dict() for memory in long_term_memories], "id"
            )

        return {
            "long_term_memories": long_term_memories,
            "candidate_notes": candidate_notes,
            "note_id_mapping": {},
            "memory_id_mapping": memory_id_mapping,
            "secretary_decision": secretary_decision,
            "core_topic": secretary_decision.get("topic", ""),
        }
