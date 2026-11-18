"""Pytest may import a hyphenated checkout's root as bare ``__init__``."""
import importlib.util
from pathlib import Path
import unittest


class RootImportTests(unittest.TestCase):
    def test_bare_root_import_does_not_require_a_package_parent(self):
        path = Path(__file__).resolve().parents[1] / '__init__.py'
        spec = importlib.util.spec_from_file_location('__init__', path, submodule_search_locations=[])
        module = importlib.util.module_from_spec(spec)
        module.__package__ = ''  # pytest's bare-root collection path
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.register))


if __name__ == '__main__':
    unittest.main()
