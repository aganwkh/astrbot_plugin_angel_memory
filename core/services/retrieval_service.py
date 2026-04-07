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
        self, event: AstrMessageEvent, query: str, precompute_vectors: bool = False
    ) -> Dict[str, Any]:
        deepmind = self.deepmind
        diagnostic_store = get_event_diagnostic_store(event)
        retrieval_diagnostic = diagnostic_store.setdefault("retrieval", {})
        time_intent = diagnostic_store.get("time_intent", {})

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
        if deepmind.memory_system:
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
                    "time_filter_passed": False,
                    "time_filter_note": "自动长期记忆检索链路未传入 start_time/end_time/time_range。",
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
        if deepmind.note_service:
            note_call_payload = {
                "note_query": note_query,
                "note_query_preview": preview_text(note_query, 160),
                "recall_count": int(deepmind.note_candidate_top_k),
                "top_k": int(deepmind.note_candidate_top_k),
                "has_precomputed_vector": note_vector is not None,
                "time_intent": time_intent,
                "time_filter_passed": False,
                "time_filter_note": "笔记检索接口当前未接收时间窗口参数。",
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
            )

            note_result_payload = summarize_note_records(candidate_notes)
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
