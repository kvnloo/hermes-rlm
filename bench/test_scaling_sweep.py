"""Offline scaling tests use a real BPE engine trained on fixed synthetic text."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.generate_scaling import EXPONENTS, generate
from bench.scaling_tokenizer import LocalTokenizer
import audit_results
import paper_matrix as matrix
from paper_proxy import Ledger


@unittest.skipUnless(importlib.util.find_spec("tokenizers"), "tokenizers is required")
class ScalingTests(unittest.TestCase):
    def setUp(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "tokenizer.json"
        raw = Tokenizer(models.BPE())
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.train_from_iterator(
            [" padding filler a x 0 .", "The source says café code.", "UNITS | entry=123 READY=0"],
            trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                               show_progress=False))
        # The actual loader must remove saved limits so no count is clipped.
        raw.enable_truncation(max_length=2)
        raw.enable_padding(length=100)
        raw.save(str(self.path))
        self.tokenizer = LocalTokenizer(self.path)

    def test_measured_manifest_and_exact_context_tokens(self):
        self.assertEqual(EXPONENTS, tuple(range(13, 21)))
        out = self.root / "out"
        stats = generate(out, self.tokenizer, exponents=(13,), variants=2)
        self.assertEqual(stats["full"], 6)
        tasks = matrix.load_tasks(out / "full.jsonl")
        self.assertEqual(len(tasks), 6)
        self.assertEqual(len({t["id"] for t in tasks}), 6)
        for task in tasks:
            context = Path(task["context_path"]).read_text(encoding="utf-8")
            self.assertEqual(self.tokenizer.count(context), 8192)
            self.assertEqual(task["size_tokens"], 8192)
            self.assertEqual(task["metadata"]["tokenizer_sha256"], self.tokenizer.sha256)
            self.assertNotIn(task["question"], context)

    def test_full_context_admits_exact_boundary_without_slicing_or_invalidating(self):
        source = self.root / "context.txt"
        text = "Café\r\n" + (" a" * 55) + "\nEnd."
        source.write_bytes(text.encode("utf-8"))
        task = dict(id="t", family="fixture", context_path=str(source), context_sha256=matrix.digest(source),
                    task_sha256="fixture", size_tokens=self.tokenizer.count(text),
                    question="Which letter?", scoring="browse_exact", gold="A")
        prompt = matrix.task_prompt(task, "base-full")
        message = text + "\n\n" + prompt
        cost = self.tokenizer.count(message)
        calls = []
        class Router:
            errors = ()
            observed = ()
            blocked = False
            def __init__(self, ledger, run, *args, **kwargs):
                self.ledger, self.run = ledger, run
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def logical_snapshot(self):
                return []
            def forward(self, body, model, role):
                calls.append(body["messages"][0]["content"])
                attempt = self.ledger.reserve(self.run, model, role)
                self.ledger.finish(attempt, 200, 0, {"prompt_tokens": cost, "completion_tokens": 1})
                return 200, b'{"choices":[{"message":{"content":"A"}}]}', {}
        from types import SimpleNamespace
        args = SimpleNamespace(ops_dir=self.root / "ops", root_model=matrix.ROOT,
            sub_model=matrix.SUB, timeout=1, disable_thinking=False, base_context_chars=1,
            max_tokens=7, base_chat_overhead_tokens=3, base_window_tokens=cost + 10,
            scorer=matrix.HERE / "paper_tasks.py")
        ledger = Ledger(self.root / "ledger.db")
        with patch.object(matrix, "Router", Router):
            fits = matrix.run_one(task, "base-full", 0, args, {}, ledger, base_tokenizer=self.tokenizer)
            args.base_window_tokens -= 1
            skips = matrix.run_one(task, "base-full", 1, args, {}, ledger, base_tokenizer=self.tokenizer)
        self.assertEqual(calls, [message])
        self.assertTrue(fits["valid"], fits)
        self.assertEqual(fits["score"], 1.0)
        self.assertEqual(fits["full_context"]["content_tokens"], cost)
        self.assertEqual(fits["truncation"], {"kind": "none", "chars": len(text),
                                             "original_chars": len(text), "truncated": False})
        self.assertEqual((skips["status"], skips["score"], skips["error"]),
                         ("skipped", None, "base_full_window_exceeded"))
        self.assertEqual(skips["accounting"]["request_attempts"], 0)
        self.assertEqual(audit_results.audit_row(fits, None, self.root)["audit_adherence"], "not_applicable_base")

    def test_dry_run_reports_full_context_eligibility_without_dispatch(self):
        source = self.root / "context.txt"
        source.write_text("Synthetic text.", encoding="utf-8")
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(dict(id="t", family="fixture", size_tokens=100,
            context_path=source.name, question="What?", gold="A", scoring="browse_exact")) + "\n",
            encoding="utf-8")
        argv = [str(manifest), str(self.root / "out.jsonl"), "--ops-dir", str(self.root / "ops"),
                "--home-template", str(self.root), "--released-source", str(self.root),
                "--released-python", sys.executable, "--fork-source", str(self.root),
                "--fork-python", sys.executable, "--nano-source", str(self.root), "--arms", "base-full",
                "--repeats", "2", "--base-tokenizer-json", str(self.path), "--base-window-tokens", "1",
                "--base-chat-overhead-tokens", "3", "--dry-run"]
        with (patch.object(matrix, "source_sha", return_value="fixture"),
              patch.object(matrix, "run_one") as run, redirect_stdout(io.StringIO()) as output):
            self.assertEqual(matrix.main(argv), 0)
            run.assert_not_called()
        previews = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(previews), 2)
        self.assertTrue(all(p["full_context"]["fits"] is False for p in previews))
        self.assertFalse((self.root / "out.jsonl").exists())
