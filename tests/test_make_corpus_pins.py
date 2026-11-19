"""Pin bench/make_corpus.py: gold answers must match what an independent reader derives from the text."""
import json
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import make_corpus  # noqa: E402

REC = re.compile(r"^\[r(\d{6})\] 2026-\d\d-\d\d (u\d+): (.*)$")


def _category(msg):
    for cat, templates in make_corpus.CATS.items():
        for t in templates:
            pat = re.escape(t).replace(re.escape("{thing}"), "(?:%s)" % "|".join(map(re.escape, make_corpus.THINGS)))
            if re.fullmatch(pat, msg):
                return cat
    return "needle" if "vault access code" in msg else None


class MakeCorpusPins(unittest.TestCase):
    def setUp(self):
        self.text, self.qs, self.meta = make_corpus.build(2000, 7)
        self.rows = []
        for i, line in enumerate(self.text.splitlines()):
            m = REC.match(line)
            self.assertIsNotNone(m, line)
            self.assertEqual(int(m.group(1)), i)
            self.rows.append((m.group(2), _category(m.group(3)), m.group(3)))
        self.gold = {q["id"]: q for q in self.qs}

    def test_size_and_meta(self):
        self.assertTrue(self.text.endswith("\n"))
        self.assertEqual(self.meta["records"], len(self.rows))
        self.assertEqual(self.meta["chars"], len(self.text) - 1)
        self.assertEqual(self.meta["seed"], 7)
        self.assertGreaterEqual(self.meta["chars"], 2000 * 4 - 200)
        self.assertLess(self.meta["chars"], 2000 * 4 + 200)

    def test_needle_is_unique_and_gold_matches(self):
        needles = [r for r in self.rows if r[1] == "needle"]
        self.assertEqual(len(needles), 1)
        code = re.search(r"is (\d-\d-\d-\d-\d);", needles[0][2]).group(1)
        self.assertEqual(self.gold["niah"]["gold"], code)
        self.assertEqual(needles[0][0], "u999")
        self.assertGreaterEqual(self.rows.index(needles[0]), len(self.rows) // 3)

    def test_needle_position_range_and_endpoints(self):
        lo, hi = set(), set()
        for seed in range(200):
            text, _, meta = make_corpus.build(175, seed)
            lines = text.splitlines()
            n = len(lines)
            self.assertEqual(n, meta["records"])
            pos = [i for i, l in enumerate(lines) if "vault access code" in l]
            self.assertEqual(len(pos), 1)
            self.assertTrue(n // 3 <= pos[0] <= n - 1, (seed, n, pos))
            lo.add(pos[0] - n // 3)
            hi.add(pos[0] - (n - 1))
        self.assertIn(0, lo)  # lower bound n//3 is reachable
        self.assertIn(0, hi)  # last record is reachable

    def test_every_other_record_is_a_known_category(self):
        self.assertTrue(all(r[1] for r in self.rows))
        self.assertEqual(sum(1 for r in self.rows if r[1] == "needle"), 1)

    def test_count_billing_gold(self):
        want = sum(1 for r in self.rows if r[1] == "billing")
        self.assertGreater(want, 0)
        self.assertEqual(self.gold["count_billing"]["gold"], want)
        self.assertEqual(self.meta["categories"], Counter(r[1] for r in self.rows if r[1] != "needle"))

    def test_top_bug_user_gold_and_ties(self):
        per = Counter(r[0] for r in self.rows if r[1] == "bug")
        top = max(per.values())
        tied = sorted(u for u, n in per.items() if n == top)
        q = self.gold["top_bug_user"]
        self.assertEqual(q["ties"], tied)
        self.assertEqual(q["gold"], f"{tied[-1]} {top}")

    def test_deterministic_and_seed_sensitive(self):
        self.assertEqual(make_corpus.build(2000, 7)[0], self.text)
        self.assertNotEqual(make_corpus.build(2000, 8)[0], self.text)

    def test_cli_writes_corpus_and_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a" / "run"
            r = subprocess.run([sys.executable, str(BENCH / "make_corpus.py"), str(out), "500", "3"],
                               capture_output=True, text=True, check=True)
            text, qs, meta = make_corpus.build(500, 3)
            self.assertEqual((out / "corpus.txt").read_text(), text)
            gold = json.loads((Path(tmp) / "a" / "run.gold.json").read_text())
            self.assertEqual(gold["questions"], qs)
            self.assertEqual(json.loads(r.stdout)["seed"], 3)
            r2 = subprocess.run([sys.executable, str(BENCH / "make_corpus.py"), str(out), "500"],
                                capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(r2.stdout)["seed"], 7)


if __name__ == "__main__":
    unittest.main()
