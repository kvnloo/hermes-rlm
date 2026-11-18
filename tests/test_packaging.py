"""Packaging and host-neutral RLM unit regressions.

No Hermes installation, credentials, or bubblewrap are needed for this suite.
The sandbox integration tests remain in rlm/tests and use pytest + Hermes.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import rlm


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_installed_distribution_ships_runtime_and_manifest(self) -> None:
        dist = importlib.metadata.distribution("hermes-rlm")
        self.assertEqual(dist.version, "0.1.0")
        installed = {str(path).replace("\\", "/") for path in (dist.files or ())}
        self.assertIn("rlm/__init__.py", installed)
        self.assertIn("rlm/kernel.py", installed)
        self.assertIn("rlm/plugin.yaml", installed)

    def test_root_manifest_matches_registered_tools(self) -> None:
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        nested = (ROOT / "rlm" / "plugin.yaml").read_text(encoding="utf-8")
        for text in (manifest, nested):
            self.assertIn("name: rlm\n", text)
            self.assertIn('version: "0.1.0"', text)
            self.assertIn("  - rlm_load\n", text)
            self.assertIn("  - rlm_exec\n", text)
        self.assertIn("manifest_version: 2\n", manifest)
        self.assertIn("api_version: 1\n", manifest)

    def test_hermes_directory_loader_entry_registers_tools(self) -> None:
        module_name = "_hermes_rlm_directory_test"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        entry = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = entry

        def cleanup() -> None:
            for name in list(sys.modules):
                if name == module_name or name.startswith(module_name + "."):
                    sys.modules.pop(name, None)

        self.addCleanup(cleanup)
        spec.loader.exec_module(entry)
        self.assertTrue(callable(entry.register))

        class RecordingContext:
            def __init__(self) -> None:
                self.tools: list[str] = []
                self.tasks: list[str] = []
                self.commands: list[str] = []
                self.bound: dict[str, tuple] = {}

            def register_tool(self, name: str, *args, **kwargs) -> None:
                self.tools.append(name)
                self.bound[name] = args[1:3]

            def register_auxiliary_task(self, name: str, **kwargs) -> None:
                self.tasks.append(name)

            def register_command(self, name: str, *args, **kwargs) -> None:
                self.commands.append(name)

        ctx = RecordingContext()
        entry.register(ctx)
        self.assertCountEqual(ctx.tools, ["rlm_load", "rlm_exec"])
        rlm_mod = sys.modules[module_name + ".rlm"]
        self.assertEqual(ctx.bound["rlm_load"], (rlm_mod.LOAD_SCHEMA, rlm_mod._load_handler))
        self.assertEqual(ctx.bound["rlm_exec"], (rlm_mod.EXEC_SCHEMA, rlm_mod._exec_handler))
        self.assertEqual(ctx.tasks, ["rlm_subcall"])
        self.assertEqual(ctx.commands, ["rlm"])


class RlmUnitTests(unittest.TestCase):
    def test_load_corpus_skips_binary_without_host_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "bin.dat").write_bytes(b"\0\1\2")
            corpus, skipped = rlm.load_corpus([str(root)], max_chars=1000)
            self.assertEqual(corpus, {"a.txt": "alpha"})
            self.assertEqual(skipped, 1)

    def test_subcall_budget_enforced_without_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = rlm.Session(
                "ci",
                subcall=lambda prompt, max_tokens: (
                    prompt.upper(), {"prompt_tokens": 1, "completion_tokens": 1}
                ),
                log_dir=Path(temp),
            )
            session.lim["max_subcalls"] = 1
            first = session._serve_llm({"id": 1, "prompts": ["abc"], "max_tokens": 20})
            self.assertEqual(first, {"type": "llm_result", "id": 1, "results": ["ABC"]})
            second = session._serve_llm({"id": 2, "prompts": ["def"], "max_tokens": 20})
            self.assertIn("budget exhausted", second["error"])
            self.assertEqual(session.stats["subcalls"], 1)
            self.assertIn('"event": "subcall"', (Path(temp) / "ci.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
