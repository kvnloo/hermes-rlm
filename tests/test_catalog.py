"""Catalog disclosure/manifest parity against the real RLM register() entry point.

Run from the repository root:
    python -m unittest discover -s tests

This intentionally needs only the Python standard library, not pytest or a running
Hermes instance. The host's plugin validator separately probes the same entry point.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RecordingContext:
    """Record every registration surface currently used by the plugin."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.auxiliary_tasks: list[str] = []
        self.commands: list[str] = []

    def register_tool(self, name: str, *args, **kwargs) -> None:
        self.tools.append(name)

    def register_auxiliary_task(self, name: str, *args, **kwargs) -> None:
        self.auxiliary_tasks.append(name)

    def register_command(self, name: str, *args, **kwargs) -> None:
        self.commands.append(name)


def manifest_list(text: str, key: str) -> list[str]:
    """Read the manifest's deliberately simple scalar-list fields without PyYAML."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == f"{key}: []":
            return []
        if line == f"{key}:":
            values = []
            for item in lines[index + 1:]:
                if not item.startswith("  - "):
                    break
                values.append(item[4:].strip())
            return values
    raise AssertionError(f"missing or unsupported manifest list: {key}")


def disclosure() -> str:
    markdown = (ROOT / "README.md").read_text(encoding="utf-8")
    return markdown.split("## What it touches\n", 1)[1].split("\n## ", 1)[0]


def disclosed_items(section: str, label: str) -> list[str]:
    row = re.search(rf"(?m)^- \*\*{re.escape(label)}\*\*: (.+)$", section)
    if row is None:
        raise AssertionError(f"README disclosure has no {label} row")
    value = row.group(1).strip()
    if value.lower().rstrip(".") == "none":
        return []
    values = re.findall(r"`([^`]+)`", value)
    if not values:
        raise AssertionError(f"README {label} row lists no names")
    return values


class CatalogTests(unittest.TestCase):
    def test_manifest_is_root_loadable_and_mirrors_nested_install(self) -> None:
        root_manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        nested_manifest = (ROOT / "rlm" / "plugin.yaml").read_text(encoding="utf-8")
        self.assertEqual(root_manifest, nested_manifest)
        self.assertIn("name: rlm", root_manifest)
        self.assertTrue((ROOT / "__init__.py").is_file())
        for field in ("provides_tools", "provides_hooks", "provides_middleware", "requires_env"):
            manifest_list(root_manifest, field)

    def test_readme_and_manifest_match_real_register(self) -> None:
        module_name = "_hermes_rlm_catalog_test"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        def cleanup() -> None:
            for key in tuple(sys.modules):
                if key == module_name or key.startswith(module_name + "."):
                    sys.modules.pop(key, None)

        self.addCleanup(cleanup)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.register))

        ctx = RecordingContext()
        module.register(ctx)

        section = disclosure()
        observed = {
            "Tools": ctx.tools,
            "Auxiliary tasks": ctx.auxiliary_tasks,
            "Commands": ctx.commands,
            "Hooks": [],
            "Middleware": [],
        }
        for label, actual in observed.items():
            self.assertEqual(
                sorted(actual), sorted(disclosed_items(section, label)),
                f"README {label} disclosure has drifted from register(ctx)",
            )
            self.assertEqual(len(actual), len(set(actual)), f"duplicate {label} registration")

        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        self.assertEqual(sorted(ctx.tools), sorted(manifest_list(manifest, "provides_tools")))
        self.assertEqual([], manifest_list(manifest, "provides_hooks"))
        self.assertEqual([], manifest_list(manifest, "provides_middleware"))

    def test_risky_behaviors_are_disclosed(self) -> None:
        section = disclosure()
        for label in ("Reads", "Writes", "Network", "Credentials", "Subprocesses"):
            self.assertRegex(section, rf"(?m)^- \*\*{label}\*\*:")
        for token in ("RLM_ALLOW_UNSANDBOXED", "RLM_LOG_DIR", "ctx.llm", "bwrap"):
            self.assertIn(token, section)


if __name__ == "__main__":
    unittest.main()
