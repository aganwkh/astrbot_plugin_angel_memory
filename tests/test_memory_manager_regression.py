import ast
import importlib.util
import logging
import sys
import re
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock


ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "angel_memory_testpkg"


def _ensure_package(name: str, path: Optional[Path] = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)] if path is not None else []
        sys.modules[name] = module
    return module


def _install_memory_manager_test_stubs() -> None:
    _ensure_package("astrbot")
    api_module = sys.modules.get("astrbot.api")
    if api_module is None:
        api_module = types.ModuleType("astrbot.api")
        api_module.logger = logging.getLogger("memory-manager-tests")
        sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot"].api = api_module

    _ensure_package(TEST_PACKAGE, ROOT)
    _ensure_package(f"{TEST_PACKAGE}.llm_memory", ROOT / "llm_memory")
    _ensure_package(f"{TEST_PACKAGE}.llm_memory.service", ROOT / "llm_memory" / "service")
    _ensure_package(f"{TEST_PACKAGE}.llm_memory.models", ROOT / "llm_memory" / "models")
    _ensure_package(f"{TEST_PACKAGE}.llm_memory.config", ROOT / "llm_memory" / "config")
    _ensure_package(f"{TEST_PACKAGE}.core", ROOT / "core")
    _ensure_package(f"{TEST_PACKAGE}.core.utils", ROOT / "core" / "utils")

    data_models_name = f"{TEST_PACKAGE}.llm_memory.models.data_models"
    if data_models_name not in sys.modules:
        data_models_module = types.ModuleType(data_models_name)

        class BaseMemory:
            pass

        class MemoryType:
            KNOWLEDGE = "knowledge"

        class ValidationError(Exception):
            pass

        data_models_module.BaseMemory = BaseMemory
        data_models_module.MemoryType = MemoryType
        data_models_module.ValidationError = ValidationError
        sys.modules[data_models_name] = data_models_module

    system_config_name = f"{TEST_PACKAGE}.llm_memory.config.system_config"
    if system_config_name not in sys.modules:
        system_config_module = types.ModuleType(system_config_name)
        system_config_module.system_config = SimpleNamespace()
        sys.modules[system_config_name] = system_config_module

    decay_policy_name = f"{TEST_PACKAGE}.llm_memory.service.memory_decay_policy"
    if decay_policy_name not in sys.modules:
        decay_policy_module = types.ModuleType(decay_policy_name)

        class MemoryDecayConfig:
            pass

        class MemoryDecayPolicy:
            def __init__(self, config=None):
                self.config = config

        decay_policy_module.MemoryDecayConfig = MemoryDecayConfig
        decay_policy_module.MemoryDecayPolicy = MemoryDecayPolicy
        sys.modules[decay_policy_name] = decay_policy_module

    query_processor_name = f"{TEST_PACKAGE}.core.utils.query_processor"
    if query_processor_name not in sys.modules:
        query_processor_module = types.ModuleType(query_processor_name)
        query_processor_module.get_query_processor = lambda: SimpleNamespace()
        sys.modules[query_processor_name] = query_processor_module

    diagnostics_name = f"{TEST_PACKAGE}.core.utils.time_diagnostics"
    if diagnostics_name not in sys.modules:
        diagnostics_module = types.ModuleType(diagnostics_name)
        diagnostics_module.preview_text = lambda *args, **kwargs: ""
        diagnostics_module.summarize_memory_records = lambda *args, **kwargs: ""
        sys.modules[diagnostics_name] = diagnostics_module


def _load_memory_manager():
    _install_memory_manager_test_stubs()
    memory_time_name = f"{TEST_PACKAGE}.core.utils.memory_time"
    if memory_time_name not in sys.modules:
        memory_time_path = ROOT / "core" / "utils" / "memory_time.py"
        memory_time_spec = importlib.util.spec_from_file_location(
            memory_time_name, memory_time_path
        )
        if memory_time_spec is None or memory_time_spec.loader is None:
            raise RuntimeError(f"failed to load module from {memory_time_path}")
        memory_time_module = importlib.util.module_from_spec(memory_time_spec)
        sys.modules[memory_time_name] = memory_time_module
        memory_time_spec.loader.exec_module(memory_time_module)
    module_name = f"{TEST_PACKAGE}.llm_memory.service.memory_manager"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    module_path = ROOT / "llm_memory" / "service" / "memory_manager.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MemoryManagerRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_consolidate_memories_deletes_archived_vectors(self):
        memory_manager_module = _load_memory_manager()
        memory_manager_cls = memory_manager_module.MemoryManager

        class DummySqlManager:
            async def consolidate_memories(self):
                return ["archived-1", "archived-2"]

        memory_index_collection = SimpleNamespace(delete=Mock())
        manager = memory_manager_cls(
            main_collection=SimpleNamespace(),
            vector_store=SimpleNamespace(),
            memory_sql_manager=DummySqlManager(),
            memory_index_collection=memory_index_collection,
        )

        await manager.consolidate_memories()

        memory_index_collection.delete.assert_called_once_with(
            ids=["archived-1", "archived-2"]
        )

    async def test_process_feedback_upserts_vectors_for_non_merge_new_memories(self):
        memory_manager_module = _load_memory_manager()
        memory_manager_cls = memory_manager_module.MemoryManager

        created_memory = SimpleNamespace(
            id="memory-new-1",
            judgment="prefers black coffee",
            tags=["user", "drink"],
        )

        class DummySqlManager:
            def __init__(self):
                self.calls = []

            async def process_feedback(self, **kwargs):
                self.calls.append(kwargs)
                return [created_memory]

            def build_vector_text(self, judgment, tags):
                return " | ".join([judgment, *tags])

        sql_manager = DummySqlManager()
        vector_store = SimpleNamespace(upsert_memory_index_rows=AsyncMock())
        memory_index_collection = object()
        manager = memory_manager_cls(
            main_collection=SimpleNamespace(),
            vector_store=vector_store,
            memory_sql_manager=sql_manager,
            memory_index_collection=memory_index_collection,
        )

        result = await manager.process_feedback(
            new_memories=[
                {
                    "type": "knowledge",
                    "judgment": "prefers black coffee",
                    "reasoning": "user said it directly",
                    "tags": ["user", "drink"],
                }
            ],
            merge_groups=None,
            memory_scope="public",
        )

        self.assertEqual(result, [created_memory])
        vector_store.upsert_memory_index_rows.assert_awaited_once_with(
            collection=memory_index_collection,
            rows=[
                {
                    "id": "memory-new-1",
                    "vector_text": "prefers black coffee | user | drink",
                }
            ],
        )

    def test_memory_record_insert_statements_have_matching_arity(self):
        source_path = ROOT / "llm_memory" / "components" / "memory_sql_manager.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

        failures = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "execute":
                continue

            sql_node = node.args[0]
            if not isinstance(sql_node, ast.Constant) or not isinstance(sql_node.value, str):
                continue

            sql_text = sql_node.value
            if "INSERT INTO memory_records(" not in sql_text:
                continue

            match = re.search(
                r"INSERT INTO memory_records\((?P<cols>.*?)\)\s*VALUES\s*\((?P<vals>.*?)\)",
                sql_text,
                re.S,
            )
            self.assertIsNotNone(match, f"failed to parse INSERT statement at line {node.lineno}")
            assert match is not None

            columns = [column.strip() for column in match.group("cols").split(",") if column.strip()]
            placeholders = re.findall(r"\?", match.group("vals"))
            params_node = node.args[1]
            if isinstance(params_node, ast.Tuple):
                param_count = len(params_node.elts)
            else:
                self.fail(
                    f"INSERT at line {node.lineno} does not pass a tuple literal as parameters"
                )

            if not (len(columns) == len(placeholders) == param_count):
                failures.append(
                    {
                        "line": node.lineno,
                        "columns": len(columns),
                        "placeholders": len(placeholders),
                        "params": param_count,
                    }
                )

        self.assertFalse(
            failures,
            "memory_records insert arity mismatch: "
            + ", ".join(
                f"line {item['line']} columns={item['columns']} placeholders={item['placeholders']} params={item['params']}"
                for item in failures
            ),
        )


if __name__ == "__main__":
    unittest.main()
