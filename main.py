"""
AstrBot Angel Memory Plugin

基于双层认知架构的AI记忆系统插件，为AstrBot提供记忆能力。
实现观察→回忆→反馈→睡眠的完整认知工作流。

采用新的懒加载+后台预初始化架构，实现极速启动和智能提供商等待。
"""

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.core.star.star_tools import StarTools
import asyncio
from datetime import datetime
import hashlib
import json
import logging
import time

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# 导入核心模块
from .core.plugin_manager import PluginManager
from .core.plugin_context import PluginContextFactory
from .tools.core_memory_remember import CoreMemoryRememberTool
from .tools.core_memory_recall import CoreMemoryRecallTool
from .tools.note_recall import NoteRecallTool
from .tools.research_tool import ResearchTool
from .core.utils.time_diagnostics import (
    analyze_recall_request,
    analyze_time_slot_classifier_trigger,
    analyze_time_intent,
    build_time_intent_from_slot,
    build_time_filter_payload,
    get_event_diagnostic_store,
    get_legal_time_slot_names,
    get_time_slot_catalog_for_prompt,
    is_low_information_followup,
    is_recall_or_review_query,
    parse_time_slot_selection_response,
    preview_text,
    summarize_raw_chat_rows,
    TIME_SLOT_CACHE_TTL_SECONDS,
    TIME_SLOT_CONFIDENCE_THRESHOLD,
)


def configure_logging_behavior():
    """统一日志行为，避免重复输出与第三方噪音日志。"""
    try:
        if isinstance(logger, logging.Logger):
            logger.propagate = False
    except Exception:
        pass

    noisy_logger_names = ["httpx", "httpcore", "urllib3"]
    for logger_name in noisy_logger_names:
        try:
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.setLevel(logging.WARNING)
            third_party_logger.propagate = False
        except Exception:
            continue


@register(
    "astrbot_plugin_angel_memory",
    "kawayiYokami",
    "天使的记忆，让astrbot拥有记忆维护系统和开箱即用的知识库检索",
    "1.4.8",
    "https://github.com/kawayiYokami/astrbot_plugin_angel_memory"
)
class AngelMemoryPlugin(Star):
    """天使记忆插件主类

    集成DeepMind记忆系统和多格式文档处理能力，为AstrBot提供完整的记忆功能。

    新架构特点：
    - 极速启动：毫秒级启动，所有耗时操作移至后台
    - 智能等待：后台自动检测提供商，有提供商时自动初始化
    - 统一实例管理：核心实例在后台异步任务中于同一事件循环创建
    - 无重复初始化：彻底解决重复初始化和实例不一致问题
    - 线程安全：避免跨线程使用异步组件的竞态条件

    插件启动后异步初始化核心实例，terminate时安全清理资源。
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        configure_logging_behavior()

        # 使用 astrbot.api 的 logger
        self.logger = logger

        # 1. 获取插件数据目录（在main.py中获取）
        data_dir = StarTools.get_data_dir("astrbot_plugin_angel_memory")
        self.logger.info(f"获取到插件数据目录: {data_dir}")

        # 2. 创建统一的PluginContext，包含所有必要资源
        self.plugin_context = PluginContextFactory.create_from_initialization(
            context, config or {}, data_dir
        )

        # 2. 核心实例占位符（将在后台初始化完成后通过ComponentFactory创建）
        self.vector_store = None
        self.cognitive_service = None
        self.deepmind = None
        self.note_service = None
        self.file_monitor = None
        # 会话ID日志提示：插件启动后每个会话只提示一次（群聊/私聊统一）
        self._conversation_id_logged_once: set[str] = set()
        self._background_tasks: set[asyncio.Task] = set()
        self._is_terminating: bool = False
        self._time_slot_followup_cache: dict[tuple[str, str, str], dict] = {}

        # 3. 在主线程获取完整配置（包含提供商信息）
        self._load_complete_config()

        # 4. 初始化插件管理器（极速启动）- 只传递PluginContext
        self.plugin_manager = PluginManager(self.plugin_context)

        # 5. 注册LLM工具
        self.llm_tools_enabled = True  # 标记LLM工具是否启用
        try:
            # 创建 ResearchTool 实例
            research_tool = ResearchTool()
            research_tool.set_context(self.context)

            self.context.add_llm_tools(
                CoreMemoryRememberTool(),
                CoreMemoryRecallTool(),
                NoteRecallTool(),
                research_tool
            )
            self.logger.info("✅ 已注册 core_memory_remember、core_memory_recall、note_recall 和 research_topic 工具。")
        except AttributeError as e:
            self.llm_tools_enabled = False
            self.logger.error(f"❌ 注册LLM工具失败，context可能不支持add_llm_tools方法: {e}", exc_info=True)
            self.logger.warning("⚠️ LLM工具功能已禁用，插件将继续以基础模式运行")
        except Exception as e:
            self.llm_tools_enabled = False
            self.logger.error(f"❌ 注册LLM工具时发生异常: {e}", exc_info=True)
            self.logger.warning("⚠️ LLM工具功能已禁用，插件将继续以基础模式运行")
            # --- 第二批次新增：初始化短时记忆滑动窗口数据库 ---
        from pathlib import Path
        self.raw_db_path = Path(data_dir) / "raw_chat_window.db"
        self._init_raw_db()
        # --- 结束新增 ---
        self.logger.info(
            f"天使记忆数据路径设置为: {self.plugin_context.get_index_dir().resolve()}"
        )
        self.logger.info(
            f"Angel Memory Plugin 实例创建完成 (提供商: {self.plugin_context.get_current_provider()}), 后台初始化已启动"
        )
        
        
                
    def _init_raw_db(self):
        import sqlite3
        with sqlite3.connect(self.raw_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_window (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_session ON chat_window(session_id)')

    def _fetch_raw_chat_records(
        self,
        session_id: str,
        limit: int | None = 15,
        start_ts: float | None = None,
        end_ts: float | None = None,
        role: str | None = None,
    ) -> list[tuple[str, str, float]]:
        import sqlite3

        with sqlite3.connect(self.raw_db_path) as conn:
            cursor = conn.cursor()
            conditions = ["session_id = ?"]
            params: list[object] = [session_id]
            if start_ts is not None and float(start_ts) > 0:
                conditions.append("timestamp >= ?")
                params.append(float(start_ts))
            if end_ts is not None and float(end_ts) > 0:
                conditions.append("timestamp <= ?")
                params.append(float(end_ts))
            if role:
                conditions.append("role = ?")
                params.append(str(role))
            query = f"""
                SELECT role, content, timestamp
                FROM chat_window
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp DESC, id DESC
            """
            if limit is not None and int(limit) > 0:
                query += "\nLIMIT ?"
                cursor.execute(query, (*params, int(limit)))
            else:
                cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

        rows.reverse()
        return rows

    def _fetch_recent_raw_chat_records(
        self, session_id: str, limit: int = 15
    ) -> list[tuple[str, str, float]]:
        return self._fetch_raw_chat_records(session_id=session_id, limit=limit)

    @staticmethod
    def _format_raw_chat_timestamp(timestamp: float) -> str:
        try:
            ts = float(timestamp or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts <= 0:
            return ""
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def _build_raw_chat_recall_block(
        self,
        rows: list[tuple[str, str, float]],
        recall_policy: dict,
    ) -> str:
        time_filter = recall_policy.get("time_filter", {}) if isinstance(recall_policy, dict) else {}
        header = str(time_filter.get("intent_type", "") or "时间窗")
        start_time = str(time_filter.get("start_time", "") or "")
        end_time = str(time_filter.get("end_time", "") or "")
        history_text = "\n".join(
            f"[{self._format_raw_chat_timestamp(timestamp)}] {str(role or '').strip()}: {str(content or '').strip()}"
            for role, content, timestamp in rows
        )
        return (
            "\n\n【🧭 原始聊天时间窗回顾结果（当前 session 真实物理记录）】\n"
            f"[回顾类型]：{header or '会话回顾'}\n"
            f"[时间范围]：{start_time or '(未指定)'} ~ {end_time or '(未指定)'}\n"
            "[硬规则]：用户此刻在追问“刚才/昨晚/昨天/前面/原话/是不是提过”的内容时，"
            "必须优先以这里的原始聊天记录为准。若此处已有内容，不要直接回答“没有记忆”。\n"
            f"{history_text}\n"
            "【原始聊天时间窗回顾结束】"
        )

    def _build_recent_fact_block(self, rows: list[tuple[str, str, float]]) -> str:
        if not rows:
            return ""
        fact_lines = "\n".join(
            f"[{self._format_raw_chat_timestamp(timestamp)}] {str(content or '').strip()}"
            for _role, content, timestamp in rows
        )
        return (
            "\n\n【🟢 近期精确事实层（当前 session，优先级高于旧长期记忆）】\n"
            "[硬规则]：以下是用户最近几轮明确说过的动作、状态、计划、对象原话。"
            "如果这些内容与更早的长期记忆冲突，优先采用这里的近期事实，禁止把新事实说偏成旧主题。\n"
            f"{fact_lines}\n"
            "【近期精确事实层结束】"
        )

    @staticmethod
    def _extract_user_message_text(event: AstrMessageEvent) -> str:
        return str(
            getattr(getattr(event, "message_obj", None), "message_str", "")
            or getattr(event, "message_str", "")
            or ""
        ).strip()

    @staticmethod
    def _classification_scope_label(recall_request: dict) -> str:
        if not isinstance(recall_request, dict):
            return "generic"
        if bool(recall_request.get("raw_chat_priority")):
            return "raw_chat"
        return "memory_recall"

    @staticmethod
    def _is_contextual_time_slot(normalized_time_range: str) -> bool:
        return str(normalized_time_range or "").strip() in {
            "just_now",
            "recent_context",
            "earlier_context",
            "last_time",
        }

    @staticmethod
    def _should_restrict_raw_chat_only(
        normalized_time_range: str,
        recall_request: dict,
    ) -> bool:
        matched_phrases = list((recall_request or {}).get("matched_phrases", []) or [])
        exact_phrase_hits = {
            "\u539f\u8bdd",
            "\u539f\u53e5",
            "\u8bf4\u4e86\u4ec0\u4e48",
            "\u8bf4\u8fc7\u4ec0\u4e48",
            "\u8bf4\u7684\u4ec0\u4e48",
            "\u8bf4\u4e86\u5565",
        }
        return bool(
            ((recall_request or {}).get("raw_chat_priority") or AngelMemoryPlugin._is_contextual_time_slot(normalized_time_range))
            and (
                AngelMemoryPlugin._is_contextual_time_slot(normalized_time_range)
                or any(phrase in exact_phrase_hits for phrase in matched_phrases)
            )
        )

    async def _resolve_time_classifier_provider(
        self,
        event: AstrMessageEvent,
    ) -> tuple[str, object | None]:
        session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
        provider_id = str(self.plugin_context.get_llm_provider_id() or "").strip()
        if not provider_id and hasattr(self.context, "get_current_chat_provider_id"):
            try:
                provider_id = str(
                    await self.context.get_current_chat_provider_id(umo=session_id)
                    or ""
                ).strip()
            except Exception as exc:
                self.logger.debug(f"[时间槽分类] 获取当前会话 provider_id 失败: {exc}")

        if not provider_id or not hasattr(self.context, "get_provider_by_id"):
            return "", None

        try:
            return provider_id, self.context.get_provider_by_id(provider_id)
        except Exception as exc:
            self.logger.warning(
                f"[时间槽分类] 解析 provider 失败 provider_id={provider_id} error={exc}"
            )
            return provider_id, None

    async def _resolve_time_slot_cache_scope(
        self,
        event: AstrMessageEvent,
    ) -> str:
        try:
            scope_name = await self.plugin_context.resolve_memory_scope_from_event(event)
            return str(scope_name or "").strip() or "public"
        except Exception:
            return "public"

    @staticmethod
    def _make_time_slot_followup_cache_key(
        session_id: str,
        scope_name: str,
        classification_scope: str,
    ) -> tuple[str, str, str]:
        return (
            str(session_id or "").strip(),
            str(scope_name or "").strip() or "public",
            str(classification_scope or "").strip() or "generic",
        )

    def _prune_time_slot_followup_cache(self) -> None:
        now_ts = time.time()
        expired_keys = [
            cache_key
            for cache_key, payload in self._time_slot_followup_cache.items()
            if now_ts - float(payload.get("cached_at", 0.0) or 0.0)
            > float(TIME_SLOT_CACHE_TTL_SECONDS)
        ]
        for cache_key in expired_keys:
            self._time_slot_followup_cache.pop(cache_key, None)

    def _get_time_slot_followup_cache_entry(
        self,
        session_id: str,
        scope_name: str,
        classification_scope: str,
    ) -> dict | None:
        self._prune_time_slot_followup_cache()
        candidate_keys = [
            self._make_time_slot_followup_cache_key(
                session_id,
                scope_name,
                classification_scope,
            )
        ]
        if classification_scope == "generic":
            candidate_keys.extend(
                [
                    self._make_time_slot_followup_cache_key(session_id, scope_name, "raw_chat"),
                    self._make_time_slot_followup_cache_key(session_id, scope_name, "memory_recall"),
                ]
            )

        candidates: list[dict] = []
        for cache_key in candidate_keys:
            payload = self._time_slot_followup_cache.get(cache_key)
            if isinstance(payload, dict):
                candidates.append(payload)

        if not candidates:
            return None
        candidates.sort(key=lambda item: float(item.get("cached_at", 0.0) or 0.0), reverse=True)
        return candidates[0]

    def _store_time_slot_followup_cache(
        self,
        session_id: str,
        scope_name: str,
        classification_scope: str,
        time_intent: dict,
        time_filter: dict,
        source_text: str,
    ) -> None:
        normalized_time_range = str(time_intent.get("normalized_time_range", "") or "").strip()
        if not normalized_time_range or not bool(time_filter.get("matched")):
            return

        cache_key = self._make_time_slot_followup_cache_key(
            session_id,
            scope_name,
            classification_scope,
        )
        self._time_slot_followup_cache[cache_key] = {
            "cached_at": time.time(),
            "session_id": session_id,
            "scope_name": scope_name,
            "classification_scope": classification_scope,
            "normalized_time_range": normalized_time_range,
            "time_intent": dict(time_intent or {}),
            "time_filter": dict(time_filter or {}),
            "source_text_preview": preview_text(source_text, 120),
            "source_text_hash": hashlib.sha1(str(source_text or "").encode("utf-8")).hexdigest(),
        }

    def _build_time_slot_classification_prompt(
        self,
        message_text: str,
        context_rows: list[tuple[str, str, float]],
        timezone_name: str,
        legal_slots: list[str],
        previous_slot: str,
        now_text: str,
    ) -> str:
        context_lines = []
        for role, content, timestamp in context_rows[-8:]:
            context_lines.append(
                {
                    "time": self._format_raw_chat_timestamp(timestamp),
                    "role": str(role or "").strip(),
                    "content": preview_text(str(content or "").strip(), 80),
                }
            )

        slot_catalog = get_time_slot_catalog_for_prompt(timezone_name=timezone_name)
        payload = {
            "role": "You are a time-slot classifier, not a chat assistant.",
            "task": "Judge which existing time slot should be used for downstream dialogue retrieval.",
            "timezone": timezone_name,
            "now": now_text,
            "current_user_input": str(message_text or ""),
            "recent_context": context_lines,
            "previous_selected_time_slot": str(previous_slot or ""),
            "legal_time_slots": legal_slots,
            "time_slot_catalog": slot_catalog,
            "classification_method": [
                "Step 1: infer relative-date semantics such as today, yesterday, last week, rolling recent period, or contextual follow-up.",
                "Step 2: infer time-of-day semantics such as early morning, morning, noon, afternoon, night, all-day, or contextual chat scope.",
                "Step 3: map the combined semantics to exactly one legal slot by semantic definition, representative examples, and adjacent boundaries.",
            ],
            "decision_rules": [
                "If the user clearly points to 今天凌晨 or 凌晨那会儿, prefer early_morning.",
                "If the user clearly points to 昨天凌晨, prefer yesterday_early_morning.",
                "If the user clearly points to 刚才 or 刚刚, prefer just_now.",
                "If the user clearly points to 上周一到上周日, map to last_weekday_0 through last_weekday_6.",
                "For low-information follow-ups such as 都聊了些什么, 那次呢, 前面那个呢, if previous_selected_time_slot is available, prefer inherit_previous.",
                "If information is insufficient, choose abstain and do not guess.",
                "Do not output time ranges and do not answer the user question.",
                "Do not choose by slot_name wording similarity.",
            ],
            "few_shots": [
                {
                    "input": "我们今天凌晨都聊了些什么？",
                    "output": {
                        "decision": "selected_time_slot",
                        "selected_time_slot": "early_morning",
                        "reason": "明确指向今天凌晨",
                    },
                },
                {
                    "input": "对了，我们今天凌晨聊了啥来着？",
                    "output": {
                        "decision": "selected_time_slot",
                        "selected_time_slot": "early_morning",
                        "reason": "明确指向今天凌晨",
                    },
                },
                {
                    "input": "昨天凌晨那会儿说了什么？",
                    "output": {
                        "decision": "selected_time_slot",
                        "selected_time_slot": "yesterday_early_morning",
                        "reason": "明确指向昨天凌晨",
                    },
                },
                {
                    "input": "刚才说到哪了？",
                    "output": {
                        "decision": "selected_time_slot",
                        "selected_time_slot": "just_now",
                        "reason": "近距离时间回指",
                    },
                },
                {
                    "input": "都聊了些什么？",
                    "output": {
                        "decision": "inherit_previous",
                        "selected_time_slot": "",
                        "reason": "低信息跟进，应继承上一轮时间槽",
                    },
                },
                {
                    "input": "之前提过吗？",
                    "output": {
                        "decision": "abstain",
                        "selected_time_slot": "",
                        "reason": "时间范围过模糊，无法稳定落槽",
                    },
                },
            ],
            "output_schema": {
                "decision": "selected_time_slot | abstain | inherit_previous",
                "selected_time_slot": "",
                "reason": "",
            },
            "rules": [
                "You must only choose from legal_time_slots.",
                "Do not invent new slot names.",
                "Return strict JSON only with no markdown fence.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    async def _resolve_final_time_intent(
        self,
        event: AstrMessageEvent,
        message_text: str,
    ) -> tuple[dict, dict, dict]:
        diagnostic_store = get_event_diagnostic_store(event)
        timezone_name = "Asia/Shanghai"
        empty_intent = build_time_intent_from_slot("", timezone_name=timezone_name).to_dict()
        empty_time_filter = build_time_filter_payload("", timezone_name=timezone_name)
        rule_time_intent = analyze_time_intent(message_text, timezone_name=timezone_name)
        recall_request = analyze_recall_request(message_text)
        trigger_payload = analyze_time_slot_classifier_trigger(
            message_text,
            recall_request=recall_request,
            time_intent=rule_time_intent,
        )
        low_info_followup = bool(trigger_payload.get("low_information_followup"))
        recall_or_review = bool(trigger_payload.get("recall_or_review"))
        should_call_classifier = bool(trigger_payload.get("should_call_classifier"))
        classification_scope = self._classification_scope_label(recall_request)
        session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
        scope_name = await self._resolve_time_slot_cache_scope(event)
        previous_cache = self._get_time_slot_followup_cache_entry(
            session_id,
            scope_name,
            classification_scope if not low_info_followup else "generic",
        )
        previous_slot = (
            str(previous_cache.get("normalized_time_range", "") or "").strip()
            if isinstance(previous_cache, dict)
            else ""
        )
        legal_slots = get_legal_time_slot_names(timezone_name=timezone_name)
        provider_id, provider = await self._resolve_time_classifier_provider(event)
        provider_available = provider is not None
        classification_status = "not_applicable"
        diagnostic_store["time_slot_trigger"] = {
            **trigger_payload,
            "provider_available": provider_available,
            "provider_id": provider_id,
            "previous_selected_time_slot": previous_slot,
            "classification_scope": classification_scope,
        }
        self.logger.info(
            "[时间槽分类][触发判定] payload="
            f"{json.dumps(diagnostic_store['time_slot_trigger'], ensure_ascii=False)}"
        )
        decision_payload = {
            "decision": "abstain",
            "selected_time_slot": "",
            "confidence": 0.0,
            "reason": "",
            "inherit_previous": False,
            "abstain": True,
            "parse_success": False,
            "is_valid_slot": False,
            "low_confidence": False,
            "error": "",
            "provider_available": provider_available,
            "provider_id": provider_id,
            "confidence_threshold": float(TIME_SLOT_CONFIDENCE_THRESHOLD),
            "low_information_followup": low_info_followup,
            "recall_or_review": recall_or_review,
            "classification_scope": classification_scope,
            "previous_selected_time_slot": previous_slot,
            "cache_ttl_seconds": int(TIME_SLOT_CACHE_TTL_SECONDS),
            "should_call_classifier": should_call_classifier,
            "trigger_reason": list(trigger_payload.get("trigger_reason", []) or []),
        }

        final_time_intent = dict(empty_intent)
        final_time_filter = dict(empty_time_filter)
        response_text = ""

        if not should_call_classifier:
            classification_status = "not_applicable"
        elif not provider_available:
            if low_info_followup and isinstance(previous_cache, dict):
                classification_status = "inherited"
                final_time_intent = dict(previous_cache.get("time_intent", {}) or {})
                final_time_filter = dict(previous_cache.get("time_filter", {}) or {})
                decision_payload.update(
                    {
                        "decision": "inherit_previous",
                        "selected_time_slot": previous_slot,
                        "reason": "provider_unavailable_followup_inherit",
                        "inherit_previous": True,
                        "abstain": False,
                        "confidence": 1.0,
                    }
                )
            else:
                classification_status = "provider_unavailable_empty"
        else:
            context_rows = []
            if session_id:
                try:
                    context_rows = await asyncio.to_thread(
                        self._fetch_recent_raw_chat_records,
                        session_id,
                        8,
                    )
                except Exception as exc:
                    self.logger.debug(f"[时间槽分类] 读取上下文失败 session={session_id} error={exc}")
            prompt = self._build_time_slot_classification_prompt(
                message_text=message_text,
                context_rows=context_rows,
                timezone_name=timezone_name,
                legal_slots=legal_slots,
                previous_slot=previous_slot,
                now_text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            try:
                llm_response = await provider.text_chat(prompt=prompt)
                response_text = self._extract_response_text(llm_response)
            except Exception as exc:
                response_text = ""
                decision_payload["error"] = f"classifier_call_failed:{exc}"
            parsed_result = parse_time_slot_selection_response(
                response_text=response_text,
                legal_slots=legal_slots,
                confidence_threshold=TIME_SLOT_CONFIDENCE_THRESHOLD,
            )
            diagnostic_store["time_slot_model_output"] = {
                "raw_response_original": str(parsed_result.raw_response_original or response_text or ""),
                "raw_response_sanitized": str(parsed_result.raw_response_sanitized or ""),
                "extraction_mode": str(parsed_result.extraction_mode or ""),
                "parse_error": str(parsed_result.parse_error or parsed_result.error or ""),
                "parsed_result": parsed_result.to_dict(),
                "parse_success": bool(parsed_result.parse_success),
            }
            self.logger.info(
                "[时间槽分类][模型原始输出] payload="
                f"{json.dumps(diagnostic_store['time_slot_model_output'], ensure_ascii=False)}"
            )
            decision_payload.update(parsed_result.to_dict())
            if parsed_result.parse_success and parsed_result.inherit_previous:
                if isinstance(previous_cache, dict):
                    classification_status = "inherited"
                    final_time_intent = dict(previous_cache.get("time_intent", {}) or {})
                    final_time_filter = dict(previous_cache.get("time_filter", {}) or {})
                    decision_payload.update(
                        {
                            "selected_time_slot": previous_slot,
                            "inherit_previous": True,
                            "abstain": False,
                            "confidence": max(float(parsed_result.confidence or 0.0), 1.0),
                        }
                    )
                else:
                    classification_status = "abstained"
                    decision_payload["abstain"] = True
                    decision_payload["error"] = "inherit_previous_without_cache"
            elif parsed_result.parse_success and not parsed_result.abstain:
                final_intent_obj = build_time_intent_from_slot(
                    parsed_result.selected_time_slot,
                    timezone_name=timezone_name,
                )
                mapped_time_filter = build_time_filter_payload(final_intent_obj)
                if mapped_time_filter.get("matched"):
                    classification_status = "classified"
                    final_time_intent = final_intent_obj.to_dict()
                    final_time_filter = mapped_time_filter
                else:
                    classification_status = "abstained"
                    decision_payload["abstain"] = True
                    decision_payload["error"] = "mapped_time_filter_unmatched"
            elif parsed_result.error == "low_confidence":
                classification_status = "low_confidence_abstained"
            elif not parsed_result.parse_success:
                classification_status = "parse_failed_abstained"
            else:
                classification_status = "abstained"

            if (
                bool(decision_payload.get("abstain"))
                and low_info_followup
                and isinstance(previous_cache, dict)
            ):
                classification_status = "inherited"
                final_time_intent = dict(previous_cache.get("time_intent", {}) or {})
                final_time_filter = dict(previous_cache.get("time_filter", {}) or {})
                decision_payload.update(
                    {
                        "decision": "inherit_previous",
                        "selected_time_slot": previous_slot,
                        "reason": str(decision_payload.get("reason", "") or "followup_inherit"),
                        "inherit_previous": True,
                        "abstain": False,
                        "confidence": max(float(decision_payload.get("confidence", 0.0) or 0.0), 1.0),
                    }
                )

        if final_time_filter.get("matched"):
            self._store_time_slot_followup_cache(
                session_id=session_id,
                scope_name=scope_name,
                classification_scope=classification_scope,
                time_intent=final_time_intent,
                time_filter=final_time_filter,
                source_text=message_text,
            )

        diagnostic_store["time_slot_legalization"] = {
            "decision": str(decision_payload.get("decision", "abstain") or "abstain"),
            "selected_time_slot": str(decision_payload.get("selected_time_slot", "") or ""),
            "slot_is_valid": bool(decision_payload.get("is_valid_slot")),
            "inherited": bool(decision_payload.get("inherit_previous")),
            "abstained": bool(decision_payload.get("abstain")),
            "normalized_time_range": str(final_time_intent.get("normalized_time_range", "") or ""),
            "classification_status": classification_status,
            "error": str(decision_payload.get("error", "") or ""),
        }
        self.logger.info(
            "[时间槽分类][合法化结果] payload="
            f"{json.dumps(diagnostic_store['time_slot_legalization'], ensure_ascii=False)}"
        )
        decision_payload["classification_status"] = classification_status
        decision_payload["final_time_intent"] = final_time_intent
        decision_payload["final_time_filter"] = final_time_filter
        decision_payload["scope_name"] = scope_name
        decision_payload["session_id"] = session_id
        diagnostic_store["time_slot_classification"] = decision_payload
        self.logger.info(
            "[时间槽分类结果] payload="
            f"{json.dumps({'session_id': session_id, 'classification_status': classification_status, 'selected_time_slot': str(decision_payload.get('selected_time_slot', '') or ''), 'confidence': float(decision_payload.get('confidence', 0.0) or 0.0), 'abstain': bool(decision_payload.get('abstain')), 'normalized_time_range': str(final_time_intent.get('normalized_time_range', '') or ''), 'time_filter': final_time_filter}, ensure_ascii=False)}"
        )
        return final_time_intent, final_time_filter, recall_request

    def _build_raw_chat_anchor(self, rows: list[tuple[str, str, float]]) -> str:
        history_text = "\n".join(
            f"{str(role or '').strip()}: {str(content or '').strip()}"
            for role, content, _timestamp in rows
        )
        return (
            "\n\n【🔴 绝对时序锚点：以下是用户近期的真实物理对话记录，不受任何框架截断影响。"
            "优先级最高！如果用户询问'刚才'、'前面'的内容，或验证其原话，必须且只能以此记录为准！】\n"
            f"{history_text}\n"
            "【锚点结束，回到当前对话状态。】"
        )

    def _remember_raw_chat_diagnostic(
        self,
        event: AstrMessageEvent,
        session_id: str,
        rows: list[tuple[str, str, float]],
    ) -> None:
        diagnostic_store = get_event_diagnostic_store(event)
        raw_chat_summary = summarize_raw_chat_rows(rows)
        raw_chat_payload = {
            "session_id": session_id,
            "row_count": len(rows),
            "summary": raw_chat_summary,
        }
        diagnostic_store["raw_chat_anchor"] = raw_chat_payload
        setattr(event, "_angel_memory_raw_chat_anchor_meta", raw_chat_payload)
        self.logger.info(
            "[时间过滤诊断][原始聊天锚点] payload="
            f"{json.dumps(raw_chat_payload, ensure_ascii=False)}"
        )

    def _remember_raw_chat_recall_diagnostic(
        self,
        event: AstrMessageEvent,
        session_id: str,
        rows: list[tuple[str, str, float]],
        recall_policy: dict,
    ) -> None:
        diagnostic_store = get_event_diagnostic_store(event)
        payload = {
            "session_id": session_id,
            "row_count": len(rows),
            "summary": summarize_raw_chat_rows(rows),
            "recall_policy": {
                "strict_time_recall": bool(recall_policy.get("strict_time_recall")),
                "prefer_raw_chat_only": bool(recall_policy.get("prefer_raw_chat_only")),
                "time_filter": recall_policy.get("time_filter", {}),
                "recall_request": recall_policy.get("recall_request", {}),
            },
        }
        diagnostic_store["raw_chat_recall"] = payload
        setattr(event, "_angel_memory_raw_chat_recall_meta", payload)
        self.logger.info(
            "[时间过滤诊断][原始聊天回顾] payload="
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _extract_response_text(response) -> str:
        if response is None:
            return ""
        return str(getattr(response, "completion_text", response) or "")

    @staticmethod
    def _looks_like_no_memory_response(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        keywords = [
            "没有记忆",
            "没找到记录",
            "没有记录",
            "不记得",
            "记不清",
            "想不起来",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _log_no_memory_response_diagnostic(
        self,
        event: AstrMessageEvent,
        response_text: str,
    ) -> None:
        diagnostic_store = get_event_diagnostic_store(event)
        angelmemory_context_raw = getattr(event, "angelmemory_context", None)
        context_data = {}
        if angelmemory_context_raw:
            try:
                context_data = json.loads(angelmemory_context_raw)
            except (json.JSONDecodeError, TypeError):
                context_data = {}

        raw_memories = context_data.get("raw_memories", []) or []
        raw_notes = context_data.get("raw_notes", []) or []
        raw_chat_anchor = diagnostic_store.get("raw_chat_anchor", {})
        raw_chat_recall = diagnostic_store.get("raw_chat_recall", {})
        recent_fact_layer = diagnostic_store.get("recent_fact_layer", {})
        final_injection = diagnostic_store.get("final_injection", {})
        no_memory_payload = {
            "response_preview": preview_text(response_text, 160),
            "raw_chat_anchor_count": int(raw_chat_anchor.get("row_count", 0) or 0),
            "raw_chat_anchor_summary": raw_chat_anchor.get("summary", {}),
            "raw_chat_recall_count": int(raw_chat_recall.get("row_count", 0) or 0),
            "raw_chat_recall_summary": raw_chat_recall.get("summary", {}),
            "recent_fact_count": int(recent_fact_layer.get("row_count", 0) or 0),
            "recent_fact_summary": recent_fact_layer.get("summary", {}),
            "raw_memory_count": len(raw_memories),
            "raw_note_count": len(raw_notes),
            "query_build": diagnostic_store.get("query_build", {}),
            "query_pipeline": diagnostic_store.get("query_pipeline", {}),
            "retrieval": diagnostic_store.get("retrieval", {}),
            "final_injection": final_injection,
            "has_any_candidate": bool(
                raw_chat_recall.get("row_count", 0)
                or recent_fact_layer.get("row_count", 0)
                or raw_chat_anchor.get("row_count", 0)
                or raw_memories
                or raw_notes
                or final_injection.get("session_memory_count", 0)
                or final_injection.get("selected_note_count", 0)
            ),
            "reason_guess": (
                "候选为空"
                if not (
                    raw_chat_recall.get("row_count", 0)
                    or recent_fact_layer.get("row_count", 0)
                    or raw_chat_anchor.get("row_count", 0)
                    or raw_memories
                    or raw_notes
                    or final_injection.get("session_memory_count", 0)
                    or final_injection.get("selected_note_count", 0)
                )
                else "存在候选但模型仍返回无记忆，需要结合上方链路日志判断被哪一步冲淡"
            ),
        }
        diagnostic_store["no_memory_response_check"] = no_memory_payload
        self.logger.info(
            "[时间过滤诊断][无记忆答复核查] payload="
            f"{json.dumps(no_memory_payload, ensure_ascii=False)}"
        )

    async def _inject_recent_raw_chat_anchor(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if not session_id:
            self.logger.debug("[短时记忆锚点] 跳过 原因=缺少有效session_id")
            return

        message_text = self._extract_user_message_text(event)
        time_intent = analyze_time_intent(message_text)
        time_filter = build_time_filter_payload(time_intent)
        recall_request = analyze_recall_request(message_text)
        matched_phrases = list(recall_request.get("matched_phrases", []) or [])
        explicit_time_window = bool(
            time_filter.get("matched") and time_filter.get("has_explicit_window")
        )
        exact_raw_chat_only = bool(
            (
                recall_request.get("raw_chat_priority")
                or time_intent.intent_type in {"刚才", "前面", "之前", "上次"}
            )
            and (
                time_intent.intent_type in {"刚才", "前面"}
                or any(
                    phrase in {"原话", "原句", "说了什么", "说过什么", "说的什么", "说了啥"}
                    for phrase in matched_phrases
                )
            )
        )
        strict_time_recall = explicit_time_window
        diagnostic_store = get_event_diagnostic_store(event)
        diagnostic_store["time_intent"] = time_intent.to_dict()
        diagnostic_store["recall_request"] = recall_request
        diagnostic_store["time_filter"] = time_filter
        diagnostic_store["raw_chat_time_recall_gate"] = {
            "raw_user_input": preview_text(message_text, 160),
            "time_intent": time_intent.to_dict(),
            "time_filter": time_filter,
            "recall_request": recall_request,
            "strict_time_recall": strict_time_recall,
            "exact_raw_chat_only": exact_raw_chat_only,
            "recall_mode": recall_mode,
            "gate_note": "明确时间窗优先于 recall phrase，recall phrase 仅作增强信号。",
        }
        self.logger.info(
            "[时间过滤诊断][原始聊天门控] payload="
            f"{json.dumps(diagnostic_store['raw_chat_time_recall_gate'], ensure_ascii=False)}"
        )

        try:
            raw_chat_recall_rows: list[tuple[str, str, float]] = []
            if strict_time_recall:
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    None,
                    float(time_filter.get("start_ts", 0.0) or 0.0),
                    float(time_filter.get("end_ts", 0.0) or 0.0),
                    None,
                )
            elif recall_request.get("raw_chat_priority"):
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    20,
                    None,
                    None,
                    None,
                )

            if not raw_chat_recall_rows and time_intent.intent_type in {"刚才", "前面"}:
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    20,
                    None,
                    None,
                    None,
                )

            if not raw_chat_recall_rows and time_intent.intent_type in {"刚才", "前面", "之前", "上次"}:
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    20,
                    None,
                    None,
                    None,
                )

            recent_user_rows = await asyncio.to_thread(
                self._fetch_raw_chat_records,
                session_id,
                6,
                None,
                None,
                "User",
            )
            rows = (
                raw_chat_recall_rows
                if raw_chat_recall_rows
                else await asyncio.to_thread(
                    self._fetch_recent_raw_chat_records, session_id, 15
                )
            )
        except Exception as e:
            self.logger.warning(
                f"[短时记忆锚点] 失败 session={session_id} error={e}",
                exc_info=True,
            )
            return

        if not rows and not recent_user_rows:
            self.logger.debug(
                f"[短时记忆锚点] 跳过 session={session_id} 原因=无物理对话记录"
            )
            return

        recall_policy = {
            "recall_request": recall_request,
            "time_filter": time_filter,
            "strict_time_recall": strict_time_recall,
            "prefer_raw_chat_only": bool(raw_chat_recall_rows and exact_raw_chat_only),
            "restrict_injection": bool(strict_time_recall or (raw_chat_recall_rows and exact_raw_chat_only)),
            "skip_notes": bool(strict_time_recall or (raw_chat_recall_rows and exact_raw_chat_only)),
            "raw_chat_recall_rows": raw_chat_recall_rows,
            "recent_fact_rows": recent_user_rows,
        }
        setattr(event, "_angel_memory_recall_policy", recall_policy)

        diagnostic_store = get_event_diagnostic_store(event)
        diagnostic_store["recall_request"] = recall_request
        diagnostic_store["time_filter"] = time_filter
        diagnostic_store["recall_policy"] = {
            "strict_time_recall": bool(recall_policy.get("strict_time_recall")),
            "prefer_raw_chat_only": bool(recall_policy.get("prefer_raw_chat_only")),
            "restrict_injection": bool(recall_policy.get("restrict_injection")),
            "skip_notes": bool(recall_policy.get("skip_notes")),
        }

        system_prompt_suffix = ""
        if raw_chat_recall_rows:
            self._remember_raw_chat_recall_diagnostic(
                event,
                session_id,
                raw_chat_recall_rows,
                recall_policy,
            )
            system_prompt_suffix += self._build_raw_chat_recall_block(
                raw_chat_recall_rows,
                recall_policy,
            )

        if recent_user_rows:
            diagnostic_store["recent_fact_layer"] = {
                "session_id": session_id,
                "row_count": len(recent_user_rows),
                "summary": summarize_raw_chat_rows(recent_user_rows),
            }
            system_prompt_suffix += self._build_recent_fact_block(recent_user_rows)

        if rows:
            self._remember_raw_chat_diagnostic(event, session_id, rows)
            system_prompt_suffix += self._build_raw_chat_anchor(rows)

        request.system_prompt = (
            f"{str(getattr(request, 'system_prompt', '') or '')}"
            f"{system_prompt_suffix}"
        )
        self.logger.info(
            f"[短时记忆锚点] 完成 session={session_id} "
            f"anchor条数={len(rows)} 回顾条数={len(raw_chat_recall_rows)} 近期事实条数={len(recent_user_rows)}"
        )

    async def _inject_recent_raw_chat_anchor_v2(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if not session_id:
            self.logger.debug("[时间槽分类] 跳过，原因=缺少有效 session_id")
            return

        message_text = self._extract_user_message_text(event)
        final_time_intent, time_filter, recall_request = await self._resolve_final_time_intent(
            event,
            message_text,
        )
        rule_time_intent = analyze_time_intent(message_text)
        normalized_time_range = str(
            final_time_intent.get("normalized_time_range", "") or ""
        ).strip()
        exact_raw_chat_only = self._should_restrict_raw_chat_only(
            normalized_time_range=normalized_time_range,
            recall_request=recall_request,
        )
        strict_time_recall = bool(time_filter.get("matched"))
        recall_mode = "time_slot_replay_full" if strict_time_recall else "standard"

        diagnostic_store = get_event_diagnostic_store(event)
        diagnostic_store["time_intent"] = final_time_intent
        diagnostic_store["time_intent_rule_preview"] = rule_time_intent.to_dict()
        diagnostic_store["recall_request"] = recall_request
        diagnostic_store["time_filter"] = time_filter
        diagnostic_store["raw_chat_time_recall_gate"] = {
            "raw_user_input": preview_text(message_text, 160),
            "time_intent": final_time_intent,
            "time_intent_rule_preview": rule_time_intent.to_dict(),
            "time_filter": time_filter,
            "recall_request": recall_request,
            "strict_time_recall": strict_time_recall,
            "exact_raw_chat_only": exact_raw_chat_only,
            "gate_note": "最终合法时间槽一旦产出，即直接驱动 strict_time_recall 与 time_filter 透传。",
        }
        self.logger.info(
            "[时间过滤诊断][原始聊天门控] payload="
            f"{json.dumps(diagnostic_store['raw_chat_time_recall_gate'], ensure_ascii=False)}"
        )

        try:
            raw_chat_recall_rows: list[tuple[str, str, float]] = []
            if strict_time_recall:
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    None,
                    float(time_filter.get("start_ts", 0.0) or 0.0),
                    float(time_filter.get("end_ts", 0.0) or 0.0),
                    None,
                )
            elif recall_request.get("raw_chat_priority"):
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    20,
                    None,
                    None,
                    None,
                )

            if not raw_chat_recall_rows and normalized_time_range in {"just_now", "recent_context"}:
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    20,
                    None,
                    None,
                    None,
                )

            if not raw_chat_recall_rows and normalized_time_range in {
                "just_now",
                "recent_context",
                "earlier_context",
                "last_time",
            }:
                raw_chat_recall_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    20,
                    None,
                    None,
                    None,
                )

            recent_user_rows = []
            if not strict_time_recall:
                recent_user_rows = await asyncio.to_thread(
                    self._fetch_raw_chat_records,
                    session_id,
                    6,
                    None,
                    None,
                    "User",
                )
            rows = (
                []
                if strict_time_recall
                else (
                    raw_chat_recall_rows
                    if raw_chat_recall_rows
                    else await asyncio.to_thread(
                        self._fetch_recent_raw_chat_records, session_id, 15
                    )
                )
            )
        except Exception as exc:
            self.logger.warning(
                f"[时间槽分类] 原始聊天锚点失败 session={session_id} error={exc}",
                exc_info=True,
            )
            return

        if not rows and not recent_user_rows and not strict_time_recall:
            self.logger.debug(
                f"[时间槽分类] 跳过 session={session_id} 原因=无物理对话记录"
            )
            return

        recall_policy = {
            "recall_mode": recall_mode,
            "recall_request": recall_request,
            "time_intent": final_time_intent,
            "time_filter": time_filter,
            "strict_time_recall": strict_time_recall,
            "disable_global_semantic_retrieval": bool(strict_time_recall),
            "raw_chat_full_replay": bool(strict_time_recall),
            "long_term_memory_full_replay": bool(strict_time_recall),
            "sort_by_timestamp": "asc" if strict_time_recall else "mixed",
            "prefer_raw_chat_only": False if strict_time_recall else bool(raw_chat_recall_rows and exact_raw_chat_only),
            "restrict_injection": bool(strict_time_recall or (raw_chat_recall_rows and exact_raw_chat_only)),
            "skip_notes": bool(strict_time_recall or (raw_chat_recall_rows and exact_raw_chat_only)),
            "raw_chat_recall_rows": raw_chat_recall_rows,
            "recent_fact_rows": recent_user_rows,
        }
        setattr(event, "_angel_memory_recall_policy", recall_policy)

        diagnostic_store["recall_policy"] = {
            "recall_mode": str(recall_policy.get("recall_mode", "") or ""),
            "strict_time_recall": bool(recall_policy.get("strict_time_recall")),
            "disable_global_semantic_retrieval": bool(recall_policy.get("disable_global_semantic_retrieval")),
            "raw_chat_full_replay": bool(recall_policy.get("raw_chat_full_replay")),
            "long_term_memory_full_replay": bool(recall_policy.get("long_term_memory_full_replay")),
            "sort_by_timestamp": str(recall_policy.get("sort_by_timestamp", "") or ""),
            "prefer_raw_chat_only": bool(recall_policy.get("prefer_raw_chat_only")),
            "restrict_injection": bool(recall_policy.get("restrict_injection")),
            "skip_notes": bool(recall_policy.get("skip_notes")),
        }
        diagnostic_store["time_slot_final_strategy"] = {
            "recall_mode": recall_mode,
            "strict_time_recall": bool(strict_time_recall),
            "time_filter_matched": bool(time_filter.get("matched")),
            "time_filter_applied": bool(strict_time_recall and time_filter.get("matched")),
            "start_ts": float(time_filter.get("start_ts", 0.0) or 0.0),
            "end_ts": float(time_filter.get("end_ts", 0.0) or 0.0),
            "disable_global_semantic_retrieval": bool(recall_policy.get("disable_global_semantic_retrieval")),
            "raw_chat_full_replay": bool(recall_policy.get("raw_chat_full_replay")),
            "long_term_memory_full_replay": bool(recall_policy.get("long_term_memory_full_replay")),
            "sort_by_timestamp": str(recall_policy.get("sort_by_timestamp", "") or ""),
            "skip_notes": bool(recall_policy.get("skip_notes")),
            "restrict_injection": bool(recall_policy.get("restrict_injection")),
            "prefer_raw_chat_only": bool(recall_policy.get("prefer_raw_chat_only")),
        }
        self.logger.info(
            "[时间槽分类][最终检索策略] payload="
            f"{json.dumps(diagnostic_store['time_slot_final_strategy'], ensure_ascii=False)}"
        )

        system_prompt_suffix = ""
        if raw_chat_recall_rows:
            self._remember_raw_chat_recall_diagnostic(
                event,
                session_id,
                raw_chat_recall_rows,
                recall_policy,
            )
            system_prompt_suffix += self._build_raw_chat_recall_block(
                raw_chat_recall_rows,
                recall_policy,
            )

        if recent_user_rows:
            diagnostic_store["recent_fact_layer"] = {
                "session_id": session_id,
                "row_count": len(recent_user_rows),
                "summary": summarize_raw_chat_rows(recent_user_rows),
            }
            system_prompt_suffix += self._build_recent_fact_block(recent_user_rows)

        if rows:
            self._remember_raw_chat_diagnostic(event, session_id, rows)
            system_prompt_suffix += self._build_raw_chat_anchor(rows)

        request.system_prompt = (
            f"{str(getattr(request, 'system_prompt', '') or '')}"
            f"{system_prompt_suffix}"
        )
        self.logger.info(
            f"[时间槽分类] 完成 session={session_id} anchor条数={len(rows)} "
            f"回顾条数={len(raw_chat_recall_rows)} 近期事实条数={len(recent_user_rows)}"
        )

    def _load_complete_config(self):
        """在主线程检查配置项"""
        try:
            config = self.plugin_context.get_all_config()
            self.logger.info(f"📋 插件配置加载完成: {list(config.keys())}")

            # 检查关键配置
            embedding_provider_id = self.plugin_context.get_embedding_provider_id()
            if embedding_provider_id:
                self.logger.info(f"✅ 检测到嵌入提供商配置: {embedding_provider_id}")
            else:
                self.logger.info(
                    "ℹ️ 未配置嵌入提供商ID，将按能力自动降级为 BM25-only（向量非必须）"
                )

            llm_provider_id = self.plugin_context.get_llm_provider_id()
            if llm_provider_id:
                self.logger.info(f"✅ 检测到LLM提供商配置: {llm_provider_id}")
            else:
                self.logger.info(
                    "ℹ️ 未配置LLM提供商ID (provider_id)，将使用基础记忆功能"
                )

            # 检查提供商可用性
            if self.plugin_context.has_providers():
                self.logger.info("✅ 检测到可用的提供商")
            else:
                self.logger.info("ℹ️ 未检测到可用提供商，将使用本地模式")

            provider_configured = bool(embedding_provider_id or llm_provider_id)
            if provider_configured and not self.plugin_context.has_providers():
                self.logger.info("ℹ️ 提供商配置已读取，但当前启动阶段注册表可能尚未就绪")
        except (AttributeError, KeyError, TypeError) as e:
            self.logger.error(f"❌ 配置检查失败: {e}")

    def update_components(self):
        """更新组件引用（在初始化完成后调用）"""
        if self.plugin_manager:
            # 从后台初始化器获取组件工厂
            component_factory = (
                self.plugin_manager.background_initializer.get_component_factory()
            )

            # 设置ComponentFactory引用到PluginContext
            self.plugin_context.set_component_factory(component_factory)

            # 获取所有组件
            components = component_factory.get_components()

            # 更新主线程组件引用
            self.vector_store = components.get("vector_store")
            self.cognitive_service = components.get("cognitive_service")
            self.deepmind = components.get("deepmind")
            self.note_service = components.get("note_service")
            self.file_monitor = components.get("file_monitor")

            # 将主线程组件设置给PluginManager
            main_components = {
                "vector_store": self.vector_store,
                "cognitive_service": self.cognitive_service,
                "deepmind": self.deepmind,
                "note_service": self.note_service,
                "file_monitor": self.file_monitor,
            }
            self.plugin_manager.set_main_thread_components(main_components)

    @filter.on_llm_request(priority=40)
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """
        LLM调用前整理记忆并注入到请求中
        """
        self.logger.debug("开始执行 on_llm_request")
        await self._log_event_persona(event)
        await self._log_group_id_once(event)
        try:
            await self._inject_recent_raw_chat_anchor_v2(event, request)

            # 检查LLM工具是否可用
            if not self.are_llm_tools_enabled():
                self.logger.debug("LLM工具未启用，跳过LLM请求处理")
                return

            # 更新组件引用
            self.update_components()
            self.logger.debug("组件引用已更新")

            # 使用共享的PluginContext处理请求
            result = await self.plugin_manager.handle_llm_request(
                event, request, self.plugin_context
            )
            self.logger.debug(f"handle_llm_request 返回结果: {result}")

            if result["status"] == "waiting":
                self.logger.info("系统正在初始化中，跳过此次LLM请求处理")
                return
            elif result["status"] == "success":
                self.logger.debug("LLM请求处理完成")
            else:
                self.logger.error(
                    f"LLM请求处理失败: {result.get('message', '未知错误')}"
                )

        except (AttributeError, ValueError, RuntimeError) as e:
            self.logger.error(f"LLM_REQUEST failed: {e}")

    async def _log_event_persona(self, event: AstrMessageEvent) -> None:
        """每条消息记录一次当前事件的人格名，便于排障。"""
        try:
            conversation_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
            persona_name = await self.plugin_context.get_event_persona_name(event)
            self.logger.info(
                f"[事件人格] 会话ID={conversation_id or '(空)'} 当前人格={persona_name or '(空)'}"
            )
        except Exception as e:
            self.logger.debug(f"事件人格日志记录失败（已忽略）: {e}")

    async def _log_group_id_once(self, event: AstrMessageEvent) -> None:
        """插件启动后每个会话仅记录一次会话ID，便于用户确认配置键。"""
        try:
            conversation_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
            if not conversation_id:
                return
            if conversation_id in self._conversation_id_logged_once:
                return

            self._conversation_id_logged_once.add(conversation_id)
            persona_name = await self.plugin_context.get_event_persona_name(event)
            resolved_scope, matched_by, matched_key = (
                self.plugin_context.resolve_memory_scope_with_source(
                    conversation_id, persona_name=persona_name
                )
            )
            match_desc = {
                "persona": "人格键",
                "conversation": "会话ID键",
                "default": "默认规则",
            }.get(matched_by, matched_by)
            self.logger.info(
                f"[会话分类提示] 当前人格={persona_name or '(空)'} 当前会话ID={conversation_id} "
                f"命中来源={match_desc} 命中键={matched_key} 目标scope={resolved_scope}。"
                f"注意：以下仅为配置示例，不会自动写入。"
                f"conversation_scope_map 示例：{{\"{conversation_id}\": \"家人\", \"{persona_name or '女友'}\": \"恋爱\"}}"
            )
        except Exception as e:
            self.logger.debug(f"会话ID日志记录失败（已忽略）: {e}")

    @filter.on_llm_response(priority=-100)
    async def on_llm_response(self, event: AstrMessageEvent, response):
        """
        LLM调用后捕获响应数据，存储到event上下文中

        Args:
            event: 消息事件
            response: LLM响应对象
        """
        self.logger.debug("开始执行 on_llm_response - 捕获响应数据")
        try:
            # 将响应数据存储到event上下文中，供after_message_sent使用
            if hasattr(event, "angelmemory_context"):
                try:
                    import time

                    context_data = json.loads(event.angelmemory_context)
                    # 添加响应数据
                    context_data["llm_response"] = {
                        "completion_text": self._extract_response_text(response),
                        "timestamp": time.time(),
                    }
                    event.angelmemory_context = json.dumps(context_data)
                    self.logger.debug("LLM响应数据已存储到event上下文")
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    self.logger.warning(f"存储响应数据失败: {e}")

            response_text = self._extract_response_text(response)
            if self._looks_like_no_memory_response(response_text):
                self._log_no_memory_response_diagnostic(event, response_text)

        except Exception as e:
            self.logger.error(f"on_llm_response failed: {e}")

    @filter.after_message_sent(priority=-100)
    async def after_message_sent(self, event: AstrMessageEvent):
        """
        消息发送后执行记忆整理，不阻塞主线程
        """
        self.logger.debug("开始执行 after_message_sent - 记忆整理")
        try:
            # --- 第二批次新增：物理级短时记忆留存（完全独立于大模型总结） ---
            session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
            # 修改 2：after_message_sent 中
            if session_id and hasattr(event, "angelmemory_context"):
                import json
                import time
                try:
                    context_data = json.loads(event.angelmemory_context)
                    user_text = getattr(event.message_obj, "message_str", getattr(event, "message_str", ""))
                    assistant_text = context_data.get("llm_response", {}).get("completion_text", "")
                    
                    if user_text and assistant_text:
                        def _write_history():
                            import sqlite3
                            with sqlite3.connect(self.raw_db_path) as conn:
                                cursor = conn.cursor()
                                now = time.time()
                                cursor.execute("INSERT INTO chat_window (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, "User", user_text, now))
                                cursor.execute("INSERT INTO chat_window (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, "Assistant", assistant_text, now))
                                cursor.execute("""
                                    DELETE FROM chat_window 
                                    WHERE session_id = ? AND id NOT IN (
                                        SELECT id FROM chat_window WHERE session_id = ? ORDER BY id DESC LIMIT 50
                                    )
                                """, (session_id, session_id))
                                conn.commit()
                        
                        await asyncio.to_thread(_write_history) # 异步调用
                except Exception as e:
                    self.logger.warning(f"短时记忆物理写入失败: {e}")
            # --- 结束新增 ---

            if self._is_terminating:
                self.logger.debug("插件正在关闭，跳过记忆整理任务提交")
                return

            # 检查LLM工具是否可用
            if not self.are_llm_tools_enabled():
                self.logger.debug("LLM工具未启用，跳过记忆整理")
                return

            # 更新组件引用
            self.update_components()

            # 检查是否有需要处理的记忆数据
            if not hasattr(event, "angelmemory_context"):
                self.logger.debug("没有记忆上下文，跳过记忆整理")
                return

            # 将记忆整理任务提交到事件循环，但不等待其完成，以避免阻塞主事件流程
            task = asyncio.create_task(
                self.plugin_manager.handle_memory_consolidation(
                    event, self.plugin_context
                )
            )
            self._track_background_task(task)
            self.logger.debug("记忆整理任务已提交至后台，不等待完成。")

        except Exception as e:
            self.logger.error(f"after_message_sent failed: {e}")

    def _track_background_task(self, task: asyncio.Task) -> None:
        """追踪后台任务，便于 terminate 阶段统一取消并等待收束。"""
        self._background_tasks.add(task)

        def _cleanup(done_task: asyncio.Task) -> None:
            self._background_tasks.discard(done_task)
            try:
                if done_task.cancelled():
                    return
                exc = done_task.exception()
                if exc is not None:
                    self.logger.error(f"后台任务异常退出: {exc}", exc_info=True)
            except Exception:
                pass

        task.add_done_callback(_cleanup)

    async def terminate(self) -> None:
        """插件卸载时的清理工作"""
        try:
            self.logger.info("Angel Memory Plugin 正在关闭...")
            self._is_terminating = True

            # 优雅收束后台任务，确保退出前最后的记忆整理完成落盘
            pending_tasks = [t for t in self._background_tasks if not t.done()]
            if pending_tasks:
                self.logger.info(f"检测到正在执行的记忆入库任务: {len(pending_tasks)} 个，等待其安全保存...")
                
                # 给后台任务最多 5 秒钟的宽限期完成数据库写入
                done, pending = await asyncio.wait(pending_tasks, timeout=15.0)
                
                if pending:
                    self.logger.warning(f"有 {len(pending)} 个任务写入超时，将被强制取消")
                    for task in pending:
                        task.cancel()
                
                self.logger.info("记忆安全落盘与后台任务收束完成")

            # 停止核心服务
            if self.plugin_manager:
                await self.plugin_manager.shutdown()

            # 获取最终状态
            status = (
                self.plugin_manager.get_status()
                if self.plugin_manager
                else {"state": "unknown"}
            )
            self.logger.info(
                f"Angel Memory Plugin 已关闭，最终状态: {status.get('state', 'unknown')}"
            )

        except (AttributeError, RuntimeError) as e:
            self.logger.error(f"Angel Memory Plugin: 插件卸载清理失败: {e}")

    def get_plugin_status(self):
        """
        获取插件状态（用于调试）

        Returns:
            dict: 插件状态信息
        """
        if not self.plugin_manager:
            return {"status": "not_initialized"}

        status = self.plugin_manager.get_status()
        # 添加PluginContext信息
        status.update(
            {
                "plugin_context": {
                    "current_provider": self.plugin_context.get_current_provider(),
                    "has_providers": self.plugin_context.has_providers(),
                    "index_dir": str(self.plugin_context.get_index_dir()),
                    "embedding_provider_id": self.plugin_context.get_embedding_provider_id(),
                    "llm_provider_id": self.plugin_context.get_llm_provider_id(),
                    "llm_tools_enabled": self.are_llm_tools_enabled(),
                }
            }
        )
        return status

    def get_plugin_context(self):
        """
        获取PluginContext实例（用于测试和调试）

        Returns:
            PluginContext: 插件上下文实例
        """
        return self.plugin_context

    def are_llm_tools_enabled(self):
        """
        检查LLM工具是否已成功启用

        Returns:
            bool: 如果LLM工具已启用返回True，否则返回False
        """
        return getattr(self, 'llm_tools_enabled', False)
