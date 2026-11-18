"""Every third-party module imported under bench/ must be declared in requirements.txt."""
import ast
from pathlib import Path
import re
import sys
import unittest

BENCH = Path(__file__).resolve().parent
PACKAGE = {"yaml": "pyyaml", "dateutil": "python-dateutil",
           "acp": "agent-client-protocol"}
HOST = {"run_agent", "rlm", "bench"}  # provided by Hermes or this repo


def imported_roots():
    local = {p.stem for p in BENCH.glob("*.py")}
    found = {}
    for path in BENCH.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root not in sys.stdlib_module_names and root not in local and root not in HOST:
                    found.setdefault(root, path.name)
    return found


class RequirementsTests(unittest.TestCase):
    def test_third_party_imports_are_declared(self):
        declared = {re.split(r"[<>=!~\[ ]", line.strip(), 1)[0].lower()
                    for line in (BENCH / "requirements.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
        missing = {root: path for root, path in imported_roots().items()
                   if PACKAGE.get(root, root).lower() not in declared}
        self.assertEqual(missing, {})


if __name__ == "__main__":
    unittest.main()
