import importlib.util
import json
import logging
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "angel_memory_recall_testpkg"


def _ensure_package(name: str, path: Path | None = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)] if path is not None else []
        sys.modules[name] = module
    return module


def _load_module(module_name: str, relative_path: str):
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_stubs() -> None:
    _ensure_package("astrbot")
    api_module = sys.modules.get("astrbot.api")
    if api_module is None:
        api_module = types.ModuleType("astrbot.api")
        api_module.logger = logging.getLogger("recall-query-tests")

        class FunctionTool:
            pass

        api_module.FunctionTool = FunctionTool
        sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot"].api = api_module

    event_module = sys.modules.get("astrbot.api.event")
    if event_module is None:
        event_module = types.ModuleType("astrbot.api.event")

        class AstrMessageEvent:
            pass

        event_module.AstrMessageEvent = AstrMessageEvent
        sys.modules["astrbot.api.event"] = event_module

    _ensure_package(TEST_PACKAGE, ROOT)
    _ensure_package(f"{TEST_PACKAGE}.core", ROOT / "core")
    _ensure_package(f"{TEST_PACKAGE}.core.utils", ROOT / "core" / "utils")
    _ensure_package(f"{TEST_PACKAGE}.tools", ROOT / "tools")
    _ensure_package(f"{TEST_PACKAGE}.llm_memory", ROOT / "llm_memory")
    _ensure_package(f"{TEST_PACKAGE}.llm_memory.service", ROOT / "llm_memory" / "service")
    _ensure_package(f"{TEST_PACKAGE}.llm_memory.models", ROOT / "llm_memory" / "models")

    session_memory_name = f"{TEST_PACKAGE}.core.session_memory"
    session_memory_module = sys.modules.get(session_memory_name)
    if session_memory_module is None:
        session_memory_module = types.ModuleType(session_memory_name)
        sys.modules[session_memory_name] = session_memory_module

    class MemoryItem:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    session_memory_module.MemoryItem = MemoryItem

    formatter_name = f"{TEST_PACKAGE}.core.utils.memory_formatter"
    if formatter_name not in sys.modules:
        formatter_module = types.ModuleType(formatter_name)

        class MemoryFormatter:
            @staticmethod
            def format_session_memories(memories):
                return "\n".join(str(getattr(memory, "judgment", "")) for memory in memories)

        formatter_module.MemoryFormatter = MemoryFormatter
        sys.modules[formatter_name] = formatter_module

    data_models_name = f"{TEST_PACKAGE}.llm_memory.models.data_models"
    if data_models_name not in sys.modules:
        data_models_module = types.ModuleType(data_models_name)
        data_models_module.BaseMemory = object
        sys.modules[data_models_name] = data_models_module


def _load_time_diagnostics():
    _install_stubs()
    return _load_module(
        f"{TEST_PACKAGE}.core.utils.time_diagnostics",
        "core/utils/time_diagnostics.py",
    )


def _load_query_processor():
    _install_stubs()
    _load_time_diagnostics()
    return _load_module(
        f"{TEST_PACKAGE}.core.utils.query_processor",
        "core/utils/query_processor.py",
    )


def _load_memory_time():
    _install_stubs()
    return _load_module(
        f"{TEST_PACKAGE}.core.utils.memory_time",
        "core/utils/memory_time.py",
    )


def _load_data_models():
    _install_stubs()
    _load_memory_time()
    sys.modules.pop(f"{TEST_PACKAGE}.llm_memory.models.data_models", None)
    return _load_module(
        f"{TEST_PACKAGE}.llm_memory.models.data_models",
        "llm_memory/models/data_models.py",
    )


def _load_core_memory_recall():
    _install_stubs()
    _load_time_diagnostics()
    return _load_module(
        f"{TEST_PACKAGE}.tools.core_memory_recall",
        "tools/core_memory_recall.py",
    )


def _load_note_service():
    _install_stubs()
    id_service_name = f"{TEST_PACKAGE}.llm_memory.service.id_service"
    id_service_module = sys.modules.get(id_service_name)
    if id_service_module is None:
        id_service_module = types.ModuleType(id_service_name)

        class IDService:
            @classmethod
            def from_plugin_context(cls, _plugin_context):
                return cls()

        id_service_module.IDService = IDService
        sys.modules[id_service_name] = id_service_module

    return _load_module(
        f"{TEST_PACKAGE}.llm_memory.service.note_service",
        "llm_memory/service/note_service.py",
    )


def _load_main_plugin():
    _install_stubs()
    _load_time_diagnostics()

    star_module = sys.modules.get("astrbot.api.star")
    if star_module is None:
        star_module = types.ModuleType("astrbot.api.star")

        class Context:
            pass

        class Star:
            def __init__(self, context):
                self.context = context

        def register(*_args, **_kwargs):
            def decorator(cls):
                return cls

            return decorator

        star_module.Context = Context
        star_module.Star = Star
        star_module.register = register
        sys.modules["astrbot.api.star"] = star_module

    provider_module = sys.modules.get("astrbot.api.provider")
    if provider_module is None:
        provider_module = types.ModuleType("astrbot.api.provider")

        class ProviderRequest:
            pass

        provider_module.ProviderRequest = ProviderRequest
        sys.modules["astrbot.api.provider"] = provider_module

    event_module = sys.modules["astrbot.api.event"]

    def _make_decorator(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    event_module.filter = types.SimpleNamespace(
        on_llm_request=_make_decorator,
        on_llm_response=_make_decorator,
        after_message_sent=_make_decorator,
    )

    star_tools_name = "astrbot.core.star.star_tools"
    if star_tools_name not in sys.modules:
        star_tools_module = types.ModuleType(star_tools_name)

        class StarTools:
            pass

        star_tools_module.StarTools = StarTools
        sys.modules[star_tools_name] = star_tools_module
        _ensure_package("astrbot.core")
        _ensure_package("astrbot.core.star")

    plugin_manager_name = f"{TEST_PACKAGE}.core.plugin_manager"
    if plugin_manager_name not in sys.modules:
        plugin_manager_module = types.ModuleType(plugin_manager_name)
        plugin_manager_module.PluginManager = type("PluginManager", (), {})
        sys.modules[plugin_manager_name] = plugin_manager_module

    plugin_context_name = f"{TEST_PACKAGE}.core.plugin_context"
    if plugin_context_name not in sys.modules:
        plugin_context_module = types.ModuleType(plugin_context_name)
        plugin_context_module.PluginContextFactory = type("PluginContextFactory", (), {})
        sys.modules[plugin_context_name] = plugin_context_module

    for module_name, class_name in [
        (f"{TEST_PACKAGE}.tools.core_memory_remember", "CoreMemoryRememberTool"),
        (f"{TEST_PACKAGE}.tools.core_memory_recall", "CoreMemoryRecallTool"),
        (f"{TEST_PACKAGE}.tools.note_recall", "NoteRecallTool"),
        (f"{TEST_PACKAGE}.tools.research_tool", "ResearchTool"),
    ]:
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            setattr(module, class_name, type(class_name, (), {}))
            sys.modules[module_name] = module

    return _load_module(
        f"{TEST_PACKAGE}.main",
        "main.py",
    )


class RecallQueryRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_build_time_intent_from_slot_supports_fixed_slot_mapping(self):
        diagnostics_module = _load_time_diagnostics()

        time_intent = diagnostics_module.build_time_intent_from_slot("last_night")
        time_filter = diagnostics_module.build_time_filter_payload(time_intent)

        self.assertTrue(time_intent.matched)
        self.assertEqual(time_intent.normalized_time_range, "last_night")
        self.assertTrue(time_filter["matched"])
        self.assertEqual(time_filter["normalized_time_range"], "last_night")

    def test_parse_time_slot_selection_response_rejects_low_confidence(self):
        diagnostics_module = _load_time_diagnostics()

        result = diagnostics_module.parse_time_slot_selection_response(
            response_text=json.dumps(
                {
                    "selected_time_slot": "last_night",
                    "confidence": 0.4,
                    "reason": "low confidence sample",
                    "inherit_previous": False,
                    "abstain": False,
                }
            ),
            legal_slots=diagnostics_module.get_legal_time_slot_names(),
            confidence_threshold=diagnostics_module.TIME_SLOT_CONFIDENCE_THRESHOLD,
        )

        self.assertTrue(result.parse_success)
        self.assertTrue(result.abstain)
        self.assertTrue(result.low_confidence)
        self.assertEqual(result.error, "low_confidence")

    def test_parse_time_slot_selection_response_rejects_non_json(self):
        diagnostics_module = _load_time_diagnostics()

        result = diagnostics_module.parse_time_slot_selection_response(
            response_text="not-json",
            legal_slots=diagnostics_module.get_legal_time_slot_names(),
            confidence_threshold=diagnostics_module.TIME_SLOT_CONFIDENCE_THRESHOLD,
        )

        self.assertFalse(result.parse_success)
        self.assertTrue(result.abstain)
        self.assertIn("json_parse_failed", result.error)

    def test_parse_time_slot_selection_response_supports_decision_schema(self):
        diagnostics_module = _load_time_diagnostics()

        result = diagnostics_module.parse_time_slot_selection_response(
            response_text=json.dumps(
                {
                    "decision": "selected_time_slot",
                    "selected_time_slot": "early_morning",
                    "reason": "明确指向今天凌晨",
                }
            ),
            legal_slots=diagnostics_module.get_legal_time_slot_names(),
            confidence_threshold=diagnostics_module.TIME_SLOT_CONFIDENCE_THRESHOLD,
        )

        self.assertTrue(result.parse_success)
        self.assertEqual(result.decision, "selected_time_slot")
        self.assertEqual(result.selected_time_slot, "early_morning")
        self.assertFalse(result.abstain)

    def test_parse_time_slot_selection_response_supports_fenced_json(self):
        diagnostics_module = _load_time_diagnostics()

        raw_response = """```json
{
  "decision": "selected_time_slot",
  "selected_time_slot": "early_morning",
  "reason": "用户明确再次询问'我们今天凌晨都聊了啥？'，结合对话历史中此前已多次回答今天凌晨的聊天内容（关于吉他），语义清晰指向当天凌晨时间段。"
}
```"""

        result = diagnostics_module.parse_time_slot_selection_response(
            response_text=raw_response,
            legal_slots=diagnostics_module.get_legal_time_slot_names(),
            confidence_threshold=diagnostics_module.TIME_SLOT_CONFIDENCE_THRESHOLD,
        )

        self.assertTrue(result.parse_success)
        self.assertEqual(result.extraction_mode, "fenced_json")
        self.assertEqual(result.selected_time_slot, "early_morning")
        self.assertFalse(result.abstain)

    def test_parse_time_slot_selection_response_supports_embedded_json(self):
        diagnostics_module = _load_time_diagnostics()

        raw_response = (
            "下面是分类结果，请直接使用：\n"
            '{"decision":"selected_time_slot","selected_time_slot":"early_morning","reason":"明确指向今天凌晨"}\n'
            "解释结束。"
        )

        result = diagnostics_module.parse_time_slot_selection_response(
            response_text=raw_response,
            legal_slots=diagnostics_module.get_legal_time_slot_names(),
            confidence_threshold=diagnostics_module.TIME_SLOT_CONFIDENCE_THRESHOLD,
        )

        self.assertTrue(result.parse_success)
        self.assertEqual(result.extraction_mode, "embedded_json")
        self.assertEqual(result.selected_time_slot, "early_morning")
        self.assertFalse(result.abstain)

    def test_analyze_recall_request_supports_chat_variants(self):
        diagnostics_module = _load_time_diagnostics()

        recall_request = diagnostics_module.analyze_recall_request("我们今天凌晨都聊了些什么？")

        self.assertTrue(recall_request["matched"])
        self.assertTrue(recall_request["raw_chat_priority"])
        self.assertIn("都聊了些什么", recall_request["matched_phrases"])

    def test_analyze_time_slot_classifier_trigger_supports_time_recall_query(self):
        diagnostics_module = _load_time_diagnostics()

        trigger_payload = diagnostics_module.analyze_time_slot_classifier_trigger(
            "我们今天凌晨都聊了些什么？"
        )

        self.assertTrue(trigger_payload["should_call_classifier"])
        self.assertIn("contains_time_expression", trigger_payload["trigger_reason"])
        self.assertIn("recall_or_review_semantics", trigger_payload["trigger_reason"])

    def test_time_slot_catalog_for_prompt_uses_semantic_cards(self):
        diagnostics_module = _load_time_diagnostics()

        catalog = diagnostics_module.get_time_slot_catalog_for_prompt()
        early_morning_card = next(
            item for item in catalog if item["slot_name"] == "early_morning"
        )

        self.assertIn("semantic_definition", early_morning_card)
        self.assertIn("representative_examples", early_morning_card)
        self.assertIn("adjacent_boundaries", early_morning_card)
        self.assertTrue(early_morning_card["representative_examples"])
        self.assertTrue(early_morning_card["adjacent_boundaries"])
        self.assertIn("00:00:00", early_morning_card["semantic_definition"])

    def test_time_slot_classification_prompt_requires_semantic_mapping(self):
        main_module = _load_main_plugin()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin._format_raw_chat_timestamp = lambda _timestamp: "2026-04-18 00:30:00"

        prompt = plugin._build_time_slot_classification_prompt(
            message_text="对了，我们今天凌晨聊了啥来着",
            context_rows=[("User", "今天凌晨那个", 0.0)],
            timezone_name="Asia/Shanghai",
            legal_slots=["early_morning", "last_night"],
            previous_slot="",
            now_text="2026-04-18 16:47:01",
        )
        payload = json.loads(prompt)

        self.assertIn("classification_method", payload)
        self.assertIn("few_shots", payload)
        self.assertIn("semantic_definition", payload["time_slot_catalog"][0])
        self.assertIn("representative_examples", payload["time_slot_catalog"][0])
        self.assertIn("adjacent_boundaries", payload["time_slot_catalog"][0])
        self.assertEqual(
            payload["output_schema"]["decision"],
            "selected_time_slot | abstain | inherit_previous",
        )

    def test_followup_cache_respects_ttl_and_narrow_scope_key(self):
        main_module = _load_main_plugin()
        diagnostics_module = _load_time_diagnostics()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin._time_slot_followup_cache = {}

        raw_chat_intent_obj = diagnostics_module.build_time_intent_from_slot("last_night")
        raw_chat_intent = raw_chat_intent_obj.to_dict()
        raw_chat_filter = diagnostics_module.build_time_filter_payload(raw_chat_intent_obj)
        memory_intent_obj = diagnostics_module.build_time_intent_from_slot("this_week")
        memory_intent = memory_intent_obj.to_dict()
        memory_filter = diagnostics_module.build_time_filter_payload(memory_intent_obj)

        plugin._store_time_slot_followup_cache(
            session_id="session-a",
            scope_name="public",
            classification_scope="raw_chat",
            time_intent=raw_chat_intent,
            time_filter=raw_chat_filter,
            source_text="昨晚那个",
        )
        plugin._store_time_slot_followup_cache(
            session_id="session-a",
            scope_name="public",
            classification_scope="memory_recall",
            time_intent=memory_intent,
            time_filter=memory_filter,
            source_text="这周那个",
        )

        raw_chat_entry = plugin._get_time_slot_followup_cache_entry(
            "session-a",
            "public",
            "raw_chat",
        )
        self.assertIsNotNone(raw_chat_entry)
        self.assertEqual(raw_chat_entry["normalized_time_range"], "last_night")

        memory_entry = plugin._get_time_slot_followup_cache_entry(
            "session-a",
            "public",
            "memory_recall",
        )
        self.assertIsNotNone(memory_entry)
        self.assertEqual(memory_entry["normalized_time_range"], "this_week")

        raw_chat_key = plugin._make_time_slot_followup_cache_key("session-a", "public", "raw_chat")
        plugin._time_slot_followup_cache[raw_chat_key]["cached_at"] = (
            time.time() - main_module.TIME_SLOT_CACHE_TTL_SECONDS - 1
        )

        self.assertIsNone(
            plugin._get_time_slot_followup_cache_entry("session-a", "public", "raw_chat")
        )
        generic_entry = plugin._get_time_slot_followup_cache_entry(
            "session-a",
            "public",
            "generic",
        )
        self.assertIsNotNone(generic_entry)
        self.assertEqual(generic_entry["normalized_time_range"], "this_week")

    async def test_resolve_final_time_intent_classifies_early_morning(self):
        main_module = _load_main_plugin()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin.logger = logging.getLogger("time-slot-tests")
        plugin._time_slot_followup_cache = {}
        plugin._fetch_recent_raw_chat_records = lambda _session_id, _limit: [
            ("User", "今天凌晨那个", 0.0),
            ("Assistant", "聊吉他那次", 0.0),
        ]
        plugin._extract_response_text = lambda response: response

        class Provider:
            async def text_chat(self, prompt=None, **_kwargs):
                payload = json.loads(prompt)
                assert payload["current_user_input"] == "我们今天凌晨都聊了些什么？"
                return json.dumps(
                    {
                        "decision": "selected_time_slot",
                        "selected_time_slot": "early_morning",
                        "reason": "明确指向今天凌晨",
                    }
                )

        class PluginContext:
            def get_llm_provider_id(self):
                return ""

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        class Context:
            async def get_current_chat_provider_id(self, umo):
                return "provider-1"

            def get_provider_by_id(self, provider_id):
                self.last_provider_id = provider_id
                return Provider()

        plugin.plugin_context = PluginContext()
        plugin.context = Context()

        event = SimpleNamespace(
            unified_msg_origin="session-a",
        )

        final_time_intent, final_time_filter, recall_request = await plugin._resolve_final_time_intent(
            event,
            "我们今天凌晨都聊了些什么？",
        )

        self.assertEqual(final_time_intent["normalized_time_range"], "early_morning")
        self.assertTrue(final_time_filter["matched"])
        self.assertEqual(final_time_filter["normalized_time_range"], "early_morning")
        self.assertTrue(recall_request["matched"])
        self.assertEqual(
            event._angel_memory_diagnostics["time_slot_classification"]["classification_status"],
            "classified",
        )

    async def test_resolve_final_time_intent_classifies_early_morning_variant(self):
        main_module = _load_main_plugin()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin.logger = logging.getLogger("time-slot-tests")
        plugin._time_slot_followup_cache = {}
        plugin._fetch_recent_raw_chat_records = lambda _session_id, _limit: []
        plugin._extract_response_text = lambda response: response

        class Provider:
            async def text_chat(self, prompt=None, **_kwargs):
                return json.dumps(
                    {
                        "decision": "selected_time_slot",
                        "selected_time_slot": "early_morning",
                        "reason": "明确指向今天凌晨",
                    }
                )

        class PluginContext:
            def get_llm_provider_id(self):
                return ""

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        class Context:
            async def get_current_chat_provider_id(self, umo):
                return "provider-1"

            def get_provider_by_id(self, provider_id):
                return Provider()

        plugin.plugin_context = PluginContext()
        plugin.context = Context()

        event = SimpleNamespace(unified_msg_origin="session-a")
        final_time_intent, final_time_filter, _ = await plugin._resolve_final_time_intent(
            event,
            "对了，我们今天凌晨聊了啥来着？",
        )

        self.assertEqual(final_time_intent["normalized_time_range"], "early_morning")
        self.assertTrue(final_time_filter["matched"])

    async def test_resolve_final_time_intent_accepts_production_fenced_json_response(self):
        main_module = _load_main_plugin()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin.logger = logging.getLogger("time-slot-tests")
        plugin._time_slot_followup_cache = {}
        plugin._fetch_recent_raw_chat_records = lambda _session_id, _limit: []
        plugin._extract_response_text = lambda response: response

        production_raw_response = """```json
{
  "decision": "selected_time_slot",
  "selected_time_slot": "early_morning",
  "reason": "用户明确再次询问'我们今天凌晨都聊了啥？'，结合对话历史中此前已多次回答今天凌晨的聊天内容（关于吉他），语义清晰指向当天凌晨时间段。"
}
```"""

        class Provider:
            async def text_chat(self, prompt=None, **_kwargs):
                return production_raw_response

        class PluginContext:
            def get_llm_provider_id(self):
                return ""

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        class Context:
            async def get_current_chat_provider_id(self, umo):
                return "provider-1"

            def get_provider_by_id(self, provider_id):
                return Provider()

        plugin.plugin_context = PluginContext()
        plugin.context = Context()

        event = SimpleNamespace(unified_msg_origin="session-a")
        final_time_intent, final_time_filter, _ = await plugin._resolve_final_time_intent(
            event,
            "是的，我们今天凌晨都聊了啥？",
        )

        self.assertEqual(final_time_intent["normalized_time_range"], "early_morning")
        self.assertTrue(final_time_filter["matched"])
        self.assertEqual(
            event._angel_memory_diagnostics["time_slot_classification"]["classification_status"],
            "classified",
        )
        self.assertEqual(
            event._angel_memory_diagnostics["time_slot_model_output"]["extraction_mode"],
            "fenced_json",
        )

    async def test_resolve_final_time_intent_classifies_yesterday_early_morning(self):
        main_module = _load_main_plugin()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin.logger = logging.getLogger("time-slot-tests")
        plugin._time_slot_followup_cache = {}
        plugin._fetch_recent_raw_chat_records = lambda _session_id, _limit: []
        plugin._extract_response_text = lambda response: response

        class Provider:
            async def text_chat(self, prompt=None, **_kwargs):
                return json.dumps(
                    {
                        "decision": "selected_time_slot",
                        "selected_time_slot": "yesterday_early_morning",
                        "reason": "明确指向昨天凌晨",
                    }
                )

        class PluginContext:
            def get_llm_provider_id(self):
                return ""

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        class Context:
            async def get_current_chat_provider_id(self, umo):
                return "provider-1"

            def get_provider_by_id(self, provider_id):
                return Provider()

        plugin.plugin_context = PluginContext()
        plugin.context = Context()

        event = SimpleNamespace(unified_msg_origin="session-a")
        final_time_intent, final_time_filter, _ = await plugin._resolve_final_time_intent(
            event,
            "昨天凌晨那会儿说了什么？",
        )

        self.assertEqual(final_time_intent["normalized_time_range"], "yesterday_early_morning")
        self.assertTrue(final_time_filter["matched"])

    async def test_resolve_final_time_intent_inherits_previous_slot_for_followup(self):
        main_module = _load_main_plugin()
        diagnostics_module = _load_time_diagnostics()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin.logger = logging.getLogger("time-slot-tests")
        plugin._time_slot_followup_cache = {}
        plugin._fetch_recent_raw_chat_records = lambda _session_id, _limit: []
        plugin._extract_response_text = lambda response: response

        inherited_intent = diagnostics_module.build_time_intent_from_slot("early_morning").to_dict()
        inherited_filter = diagnostics_module.build_time_filter_payload(
            diagnostics_module.build_time_intent_from_slot("early_morning")
        )
        plugin._store_time_slot_followup_cache(
            session_id="session-a",
            scope_name="public",
            classification_scope="raw_chat",
            time_intent=inherited_intent,
            time_filter=inherited_filter,
            source_text="我们今天凌晨都聊了些什么？",
        )

        class Provider:
            async def text_chat(self, prompt=None, **_kwargs):
                return json.dumps(
                    {
                        "decision": "inherit_previous",
                        "selected_time_slot": "",
                        "reason": "低信息跟进",
                    }
                )

        class PluginContext:
            def get_llm_provider_id(self):
                return ""

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        class Context:
            async def get_current_chat_provider_id(self, umo):
                return "provider-1"

            def get_provider_by_id(self, provider_id):
                self.last_provider_id = provider_id
                return Provider()

        plugin.plugin_context = PluginContext()
        plugin.context = Context()

        event = SimpleNamespace(
            unified_msg_origin="session-a",
        )

        final_time_intent, final_time_filter, _ = await plugin._resolve_final_time_intent(
            event,
            "都聊了些什么？",
        )

        self.assertEqual(final_time_intent["normalized_time_range"], "early_morning")
        self.assertTrue(final_time_filter["matched"])
        self.assertEqual(
            event._angel_memory_diagnostics["time_slot_classification"]["classification_status"],
            "inherited",
        )

    async def test_resolve_final_time_intent_allows_abstain_with_full_model_logs(self):
        main_module = _load_main_plugin()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin.logger = logging.getLogger("time-slot-tests")
        plugin._time_slot_followup_cache = {}
        plugin._fetch_recent_raw_chat_records = lambda _session_id, _limit: []
        plugin._extract_response_text = lambda response: response

        class Provider:
            async def text_chat(self, prompt=None, **_kwargs):
                return json.dumps(
                    {
                        "decision": "abstain",
                        "selected_time_slot": "",
                        "reason": "时间范围过模糊",
                    }
                )

        class PluginContext:
            def get_llm_provider_id(self):
                return ""

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        class Context:
            async def get_current_chat_provider_id(self, umo):
                return "provider-1"

            def get_provider_by_id(self, provider_id):
                return Provider()

        plugin.plugin_context = PluginContext()
        plugin.context = Context()

        event = SimpleNamespace(unified_msg_origin="session-a")
        final_time_intent, final_time_filter, _ = await plugin._resolve_final_time_intent(
            event,
            "之前提过吗？",
        )

        self.assertFalse(final_time_filter["matched"])
        self.assertEqual(final_time_intent["normalized_time_range"], "")
        self.assertEqual(
            event._angel_memory_diagnostics["time_slot_classification"]["classification_status"],
            "abstained",
        )
        self.assertIn("time_slot_model_output", event._angel_memory_diagnostics)

    async def test_inject_recent_raw_chat_anchor_uses_strict_time_window_when_slot_selected(self):
        main_module = _load_main_plugin()
        diagnostics_module = _load_time_diagnostics()
        plugin = main_module.AngelMemoryPlugin.__new__(main_module.AngelMemoryPlugin)
        plugin.logger = logging.getLogger("time-slot-tests")
        plugin._time_slot_followup_cache = {}

        early_morning_intent = diagnostics_module.build_time_intent_from_slot("early_morning").to_dict()
        early_morning_filter = diagnostics_module.build_time_filter_payload(
            diagnostics_module.build_time_intent_from_slot("early_morning")
        )
        raw_chat_calls = []

        async def _resolve_final_time_intent(_event, _message_text):
            return (
                early_morning_intent,
                early_morning_filter,
                {
                    "matched": True,
                    "matched_phrases": ["都聊了些什么"],
                    "raw_chat_priority": True,
                    "recent_fact_priority": True,
                    "time_expression_priority": True,
                    "time_intent": early_morning_intent,
                },
            )

        def _fetch_raw_chat_records(session_id, limit, start_ts, end_ts, role):
            raw_chat_calls.append(
                {
                    "session_id": session_id,
                    "limit": limit,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "role": role,
                }
            )
            if role == "User":
                return []
            return [("User", "今天凌晨那个", 1.0), ("Assistant", "聊吉他", 2.0)]

        def _fetch_recent_raw_chat_records(*_args, **_kwargs):
            raise AssertionError("strict_time_recall should not fallback to recent raw chat anchor")

        plugin._resolve_final_time_intent = _resolve_final_time_intent
        plugin._extract_user_message_text = lambda _event: "我们今天凌晨都聊了些什么？"
        plugin._fetch_raw_chat_records = _fetch_raw_chat_records
        plugin._fetch_recent_raw_chat_records = _fetch_recent_raw_chat_records
        plugin._should_restrict_raw_chat_only = lambda normalized_time_range, recall_request: False
        plugin._format_raw_chat_timestamp = lambda _timestamp: "2026-04-18 00:30:00"
        plugin._remember_raw_chat_recall_diagnostic = lambda *args, **kwargs: None
        plugin._build_raw_chat_recall_block = lambda rows, recall_policy: "RAW_CHAT_RECALL_BLOCK"
        plugin._build_recent_fact_block = lambda rows: "RECENT_FACT_BLOCK"
        plugin._remember_raw_chat_diagnostic = lambda *args, **kwargs: None
        plugin._build_raw_chat_anchor = lambda rows: "RAW_CHAT_ANCHOR_BLOCK"

        event = SimpleNamespace(unified_msg_origin="session-a")
        request = SimpleNamespace(system_prompt="")

        await plugin._inject_recent_raw_chat_anchor_v2(event, request)

        self.assertTrue(event._angel_memory_recall_policy["strict_time_recall"])
        self.assertEqual(event._angel_memory_recall_policy["recall_mode"], "time_slot_replay_full")
        self.assertTrue(event._angel_memory_recall_policy["disable_global_semantic_retrieval"])
        self.assertTrue(event._angel_memory_recall_policy["raw_chat_full_replay"])
        self.assertTrue(event._angel_memory_recall_policy["long_term_memory_full_replay"])
        self.assertEqual(event._angel_memory_recall_policy["sort_by_timestamp"], "asc")
        self.assertTrue(event._angel_memory_recall_policy["skip_notes"])
        self.assertTrue(event._angel_memory_recall_policy["restrict_injection"])
        self.assertTrue(event._angel_memory_recall_policy["time_filter"]["matched"])
        self.assertFalse(event._angel_memory_recall_policy["prefer_raw_chat_only"])
        self.assertEqual(
            event._angel_memory_recall_policy["time_filter"]["normalized_time_range"],
            "early_morning",
        )
        self.assertIsNone(raw_chat_calls[0]["limit"])
        self.assertEqual(raw_chat_calls[0]["start_ts"], early_morning_filter["start_ts"])
        self.assertEqual(raw_chat_calls[0]["end_ts"], early_morning_filter["end_ts"])
        self.assertIn("RAW_CHAT_RECALL_BLOCK", request.system_prompt)
        self.assertTrue(
            event._angel_memory_diagnostics["time_slot_final_strategy"]["time_filter_applied"]
        )
        self.assertEqual(
            event._angel_memory_diagnostics["time_slot_final_strategy"]["recall_mode"],
            "time_slot_replay_full",
        )

    async def test_core_memory_recall_falls_back_when_only_inactive_hits(self):
        recall_module = _load_core_memory_recall()
        tool = recall_module.CoreMemoryRecallTool()
        tool.__post_init__()

        inactive_memory = SimpleNamespace(
            id="memory-1",
            memory_type=SimpleNamespace(value="knowledge"),
            judgment="昨晚凌晨洗完衣服并互道晚安，提醒第二天有满课",
            reasoning="回忆命中，但不是主动记忆",
            tags=["昨晚", "满课"],
            strength=9,
            is_active=False,
            created_at=1712851200.0,
            similarity=0.92,
        )

        class PluginContext:
            def __init__(self, memory_runtime):
                self._memory_runtime = memory_runtime

            def get_component(self, name):
                if name == "memory_runtime":
                    return self._memory_runtime
                return None

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        memory_runtime = SimpleNamespace(
            comprehensive_recall=AsyncMock(return_value=[inactive_memory]),
        )
        event = SimpleNamespace(
            plugin_context=PluginContext(memory_runtime),
            message_obj=SimpleNamespace(message_str="昨晚你还记得吗"),
        )

        result = await tool.run(
            event=event,
            limit=1,
            query="今天上午满课 对啊 下午",
            time_range="last_night",
            original_user_text="昨晚你还记得吗",
            original_query="昨晚 今天上午满课 对啊 下午",
        )

        self.assertIn("命中了相关记忆", result)
        kwargs = memory_runtime.comprehensive_recall.await_args.kwargs
        self.assertTrue(kwargs["time_filter"]["matched"])
        self.assertEqual(kwargs["time_filter"]["normalized_time_range"], "last_night")
        diagnostic = event._angel_memory_diagnostics["tool_recall"]
        self.assertEqual(diagnostic["total_hits"], 1)
        self.assertEqual(diagnostic["active_hits"], 0)
        self.assertEqual(diagnostic["final_returned_hits"], 1)

    async def test_core_memory_recall_uses_event_recall_policy_time_filter(self):
        recall_module = _load_core_memory_recall()
        diagnostics_module = _load_time_diagnostics()
        tool = recall_module.CoreMemoryRecallTool()
        tool.__post_init__()

        class PluginContext:
            def __init__(self, memory_runtime):
                self._memory_runtime = memory_runtime

            def get_component(self, name):
                if name == "memory_runtime":
                    return self._memory_runtime
                return None

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        memory_runtime = SimpleNamespace(
            comprehensive_recall=AsyncMock(return_value=[]),
        )
        inherited_time_intent = diagnostics_module.build_time_intent_from_slot("last_night")
        inherited_time_filter = diagnostics_module.build_time_filter_payload(inherited_time_intent)
        event = SimpleNamespace(
            plugin_context=PluginContext(memory_runtime),
            message_obj=SimpleNamespace(message_str="你还记得吗"),
            _angel_memory_recall_policy={
                "time_intent": inherited_time_intent.to_dict(),
                "time_filter": inherited_time_filter,
                "strict_time_recall": True,
            },
        )

        await tool.run(
            event=event,
            limit=1,
            query="昨晚聊了什么",
            time_range="",
            original_user_text="昨晚那个",
            original_query="昨晚聊了什么",
        )

        kwargs = memory_runtime.comprehensive_recall.await_args.kwargs
        self.assertIsNotNone(kwargs["time_filter"])
        self.assertTrue(kwargs["time_filter"]["matched"])
        self.assertEqual(kwargs["time_filter"]["normalized_time_range"], "last_night")

    async def test_core_memory_recall_full_replay_uses_time_window_listing(self):
        recall_module = _load_core_memory_recall()
        diagnostics_module = _load_time_diagnostics()
        tool = recall_module.CoreMemoryRecallTool()
        tool.__post_init__()

        class PluginContext:
            def __init__(self, memory_runtime):
                self._memory_runtime = memory_runtime

            def get_component(self, name):
                if name == "memory_runtime":
                    return self._memory_runtime
                return None

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        memory_a = SimpleNamespace(
            id="memory-a",
            memory_type=SimpleNamespace(value="event"),
            judgment="凌晨先聊了吉他借来玩",
            reasoning="第一条长期记忆",
            tags=["吉他"],
            strength=1,
            is_active=False,
            created_at=20.0,
            event_start_ts=10.0,
            event_end_ts=10.0,
        )
        memory_b = SimpleNamespace(
            id="memory-b",
            memory_type=SimpleNamespace(value="event"),
            judgment="后来又聊到木吉他吃灰",
            reasoning="第二条长期记忆",
            tags=["木吉他"],
            strength=1,
            is_active=True,
            created_at=30.0,
            event_start_ts=15.0,
            event_end_ts=15.0,
        )
        memory_runtime = SimpleNamespace(
            list_memories_in_time_window=AsyncMock(return_value=[memory_b, memory_a]),
            comprehensive_recall=AsyncMock(return_value=[]),
        )
        early_morning_intent = diagnostics_module.build_time_intent_from_slot("early_morning")
        early_morning_filter = diagnostics_module.build_time_filter_payload(early_morning_intent)
        event = SimpleNamespace(
            plugin_context=PluginContext(memory_runtime),
            message_obj=SimpleNamespace(message_str="我们今天凌晨都聊了些什么？"),
            _angel_memory_recall_policy={
                "recall_mode": "time_slot_replay_full",
                "disable_global_semantic_retrieval": True,
                "raw_chat_full_replay": True,
                "long_term_memory_full_replay": True,
                "sort_by_timestamp": "asc",
                "time_intent": early_morning_intent.to_dict(),
                "time_filter": early_morning_filter,
                "strict_time_recall": True,
            },
        )

        result = await tool.run(
            event=event,
            limit=1,
            query="我们今天凌晨都聊了些什么？",
            time_range="",
            original_user_text="我们今天凌晨都聊了些什么？",
            original_query="我们今天凌晨都聊了些什么？",
        )

        memory_runtime.comprehensive_recall.assert_not_awaited()
        kwargs = memory_runtime.list_memories_in_time_window.await_args.kwargs
        self.assertEqual(kwargs["sort_order"], "asc")
        self.assertTrue(kwargs["time_filter"]["matched"])
        self.assertLess(result.find("凌晨先聊了吉他借来玩"), result.find("后来又聊到木吉他吃灰"))
        self.assertEqual(
            event._angel_memory_diagnostics["tool_recall"]["final_returned_mode"],
            "time_slot_replay_full",
        )

    async def test_core_memory_recall_does_not_fallback_to_rule_time_classification(self):
        recall_module = _load_core_memory_recall()
        tool = recall_module.CoreMemoryRecallTool()
        tool.__post_init__()

        class PluginContext:
            def __init__(self, memory_runtime):
                self._memory_runtime = memory_runtime

            def get_component(self, name):
                if name == "memory_runtime":
                    return self._memory_runtime
                return None

            async def resolve_memory_scope_from_event(self, _event):
                return "public"

        memory_runtime = SimpleNamespace(
            comprehensive_recall=AsyncMock(return_value=[]),
        )
        event = SimpleNamespace(
            plugin_context=PluginContext(memory_runtime),
            message_obj=SimpleNamespace(message_str="昨晚那个"),
        )

        await tool.run(
            event=event,
            limit=1,
            query="昨晚那个",
            time_range="",
            original_user_text="昨晚那个",
            original_query="昨晚那个",
        )

        kwargs = memory_runtime.comprehensive_recall.await_args.kwargs
        self.assertIsNone(kwargs["time_filter"])

    def test_query_processor_handles_dict_context_and_keeps_postprocessing(self):
        query_processor_module = _load_query_processor()
        processor = query_processor_module.QueryProcessor()
        event = SimpleNamespace(
            angelheart_context={
                "secretary_decision": {
                    "entities": ["Angel", "Alice"],
                    "facts": ["Angel 今天上午满课"],
                    "keywords": ["Angel"],
                    "persona_name": "Angel",
                    "alias": ["天使", "Angel"],
                }
            }
        )

        final_query = processor.process_query(
            "Angel 今天上午满课 对啊 下午",
            event,
            query_kind="memory",
        )

        self.assertNotIn("Angel", final_query)
        diagnostic = event._angel_memory_diagnostics["query_pipeline"]["memory"]
        self.assertTrue(diagnostic["assistant_filter_applied"])
        self.assertTrue(diagnostic["used_rag_query"])
        self.assertTrue(diagnostic["preprocess_skipped"])
        context_diagnostic = event._angel_memory_diagnostics["query_pipeline"]["angelheart_context"]
        self.assertEqual(context_diagnostic["angelheart_context_type"], "dict")
        self.assertEqual(context_diagnostic["parse_mode"], "dict")

    def test_time_diagnostics_supports_explicit_time_priority(self):
        diagnostics_module = _load_time_diagnostics()

        time_intent = diagnostics_module.analyze_time_intent("上周四那个事你还记得吗")
        time_filter = diagnostics_module.build_time_filter_payload(time_intent)
        recall_request = diagnostics_module.analyze_recall_request("上周四那个事你还记得吗")

        self.assertTrue(time_filter["matched"])
        self.assertEqual(time_intent.intent_type, "上周四")
        self.assertEqual(time_intent.normalized_time_range, "last_weekday_3")
        self.assertTrue(recall_request["matched"])
        self.assertTrue(recall_request["time_expression_priority"])

    async def test_note_service_accepts_time_filter_but_marks_unsupported(self):
        note_service_module = _load_note_service()

        class PluginContext:
            def get_component(self, name):
                if name == "memory_sql_manager":
                    return object()
                return None

        service = note_service_module.NoteService(plugin_context=PluginContext())
        service._search_notes_v2 = AsyncMock(
            return_value=[
                {
                    "id": "note-1",
                    "content": "昨晚的笔记",
                    "metadata": {"updated_at": 0},
                    "similarity": 0.8,
                }
            ]
        )

        result = await service.search_notes_by_top_k(
            query="昨晚",
            recall_count=5,
            top_k=1,
            time_filter={"matched": True, "normalized_time_range": "last_night"},
        )

        self.assertEqual(len(result), 1)
        kwargs = service._search_notes_v2.await_args.kwargs
        self.assertTrue(kwargs["time_filter"]["matched"])
        self.assertEqual(kwargs["time_filter"]["normalized_time_range"], "last_night")

    def test_memory_time_prefers_event_window_over_created_at(self):
        memory_time_module = _load_memory_time()
        data_models_module = _load_data_models()

        memory = data_models_module.BaseMemory(
            memory_type=data_models_module.MemoryType.EVENT,
            judgment="昨晚一起聊了满课安排",
            reasoning="事件发生在昨晚，但今天下午才总结入库",
            tags=["昨晚", "满课"],
            created_at=1712912400.0,
            event_start_ts=1712858400.0,
            event_end_ts=1712862000.0,
            event_time_confidence="exact",
        )
        time_filter = {
            "matched": True,
            "start_ts": 1712854800.0,
            "end_ts": 1712865599.0,
        }

        match = memory_time_module.classify_memory_time_match(memory, time_filter)

        self.assertTrue(match["matched"])
        self.assertEqual(match["match_type"], "event_exact")

    def test_memory_time_created_at_only_is_not_exact_match(self):
        memory_time_module = _load_memory_time()
        data_models_module = _load_data_models()

        memory = data_models_module.BaseMemory(
            memory_type=data_models_module.MemoryType.EVENT,
            judgment="今天下午才写入的旧记忆",
            reasoning="只有写入时间，没有事件时间",
            tags=["回忆"],
            created_at=1712858400.0,
        )
        time_filter = {
            "matched": True,
            "start_ts": 1712854800.0,
            "end_ts": 1712865599.0,
        }

        match = memory_time_module.classify_memory_time_match(memory, time_filter)

        self.assertEqual(match["match_type"], "created_at_only")


if __name__ == "__main__":
    unittest.main()
