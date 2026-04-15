import importlib.util
import logging
import sys
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


class RecallQueryRegressionTests(unittest.IsolatedAsyncioTestCase):
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
            time_range="",
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
