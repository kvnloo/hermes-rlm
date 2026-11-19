"""Pin the bench workflow's configuration (stdlib only, so it runs with no deps).

Without the install step and BENCH_REQUIRE_DEPS the bench job would pass on
skipped tests. This pins the config; it does not prove the job runs on Actions.
"""
import re
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "bench.yml"


class BenchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.text = WORKFLOW.read_text()

    def test_runs_on_pull_request(self):
        self.assertRegex(self.text, r"(?m)^on:\n  pull_request:\n(?!    \S)")

    def test_installs_bench_requirements(self):
        self.assertRegex(self.text, r"(?m)^\s+run: python -m pip install -r bench/requirements\.txt$")

    def test_discover_step_requires_deps(self):
        step = re.search(r"(?m)^      - name: [^\n]*\n(?:(?!      - ).*\n)*?\s+run: python -m unittest discover -s bench$", self.text)
        self.assertIsNotNone(step)
        self.assertRegex(step.group(0), r"(?m)^        env:\n          BENCH_REQUIRE_DEPS: '1'$")


if __name__ == "__main__":
    unittest.main()
