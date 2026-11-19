"""Pin run_bench.kilo_credentials and run_nano (yaml stubbed, subprocess mocked; no network or user home)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import run_bench  # noqa: E402

YAML = {"yaml": SimpleNamespace(safe_load=json.loads)}  # config written as JSON, a YAML subset


class KiloCredentials(unittest.TestCase):
    def _creds(self, cfg, provider="p", env=None):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.yaml").write_text(json.dumps(cfg))
            with mock.patch.dict(sys.modules, YAML), mock.patch.object(run_bench, "HERMES_HOME", Path(d)), \
                    mock.patch.dict(run_bench.os.environ, env or {}, clear=True):
                return run_bench.kilo_credentials(provider)

    def test_api_key_beats_key_env(self):
        cfg = {"providers": {"p": {"base_url": "http://b", "api_key": "K", "key_env": "E"}}}
        self.assertEqual(self._creds(cfg, env={"E": "fromenv"}), ("http://b", "K"))

    def test_key_env_fallback_and_empty_when_unset(self):
        cfg = {"providers": {"p": {"base_url": "http://b", "key_env": "E"}}}
        self.assertEqual(self._creds(cfg, env={"E": "fromenv"}), ("http://b", "fromenv"))
        self.assertEqual(self._creds(cfg), ("http://b", ""))
        self.assertEqual(self._creds({"providers": {"p": {"base_url": "u"}}}), ("u", ""))

    def test_missing_provider_or_null_providers_raises_keyerror(self):
        for cfg in ({"providers": {"other": {"base_url": "x"}}}, {"providers": None}, {}):
            with self.assertRaises(KeyError):
                self._creds(cfg)

    def test_only_the_requested_provider_is_used(self):
        cfg = {"providers": {"a": {"base_url": "A", "api_key": "ka"}, "b": {"base_url": "B", "api_key": "kb"}}}
        self.assertEqual(self._creds(cfg, provider="b"), ("B", "kb"))


def _cp(out="", err=""):
    return subprocess.CompletedProcess([], 0, out, err)


class RunNano(unittest.TestCase):
    def _run(self, proc):
        args = SimpleNamespace(provider="prov", model="mod", timeout=7)
        with mock.patch.object(run_bench, "kilo_credentials", return_value=("http://b", "K")) as kc, \
                mock.patch.object(run_bench.subprocess, "run", return_value=proc) as r, \
                mock.patch.object(run_bench.time, "monotonic", side_effect=[10.0, 11.23456]), \
                mock.patch.dict(run_bench.os.environ, {"INHERITED": "yes"}):
            row = run_bench.run_nano("PROMPT", "/corp", args, Path("/rd"))
        kc.assert_called_once_with("prov")
        return row, r

    def test_invocation_argv_cwd_timeout_and_env(self):
        _, r = self._run(_cp(json.dumps({"answer": "a"})))
        cmd = r.call_args.args[0]
        self.assertEqual(cmd, [str(run_bench.NANO / ".venv/bin/python"), str(run_bench.HERE / "nano_client.py"), "PROMPT"])
        kw = r.call_args.kwargs
        self.assertEqual((kw["cwd"], kw["timeout"], kw["capture_output"], kw["text"]), ("/corp", 7, True, True))
        env = kw["env"]
        self.assertEqual({k: env[k] for k in ("NANO_BASE_URL", "NANO_API_KEY", "NANO_MODEL", "NANO_RLM_BIN", "RLM_HOME")},
                         {"NANO_BASE_URL": "http://b", "NANO_API_KEY": "K", "NANO_MODEL": "mod",
                          "NANO_RLM_BIN": str(run_bench.NANO / ".venv/bin/rlm"), "RLM_HOME": "/rd/nano-home"})
        self.assertEqual(env["INHERITED"], "yes")

    def test_last_stdout_line_is_parsed_and_row_shape(self):
        snap = {"usage": {"t": 1}, "metrics": {"m": 2}, "last_stop_reason": "done"}
        out = "noise\n" + json.dumps({"answer": "x" * 2500 + "TAIL", "snapshot": snap}) + "\n"
        row, _ = self._run(_cp(out))
        self.assertEqual(set(row), {"answer", "wall_s", "nano_usage", "nano_metrics", "nano_stop"})
        self.assertEqual(row["answer"], ("x" * 2500 + "TAIL")[-2000:])
        self.assertEqual((row["nano_usage"], row["nano_metrics"], row["nano_stop"]), ({"t": 1}, {"m": 2}, "done"))
        self.assertEqual(row["wall_s"], 1.23)  # monotonic elapsed, 2dp

    def test_usage_falls_back_to_token_usage_then_empty(self):
        row, _ = self._run(_cp(json.dumps({"answer": "a", "snapshot": {"token_usage": {"u": 3}}})))
        self.assertEqual(row["nano_usage"], {"u": 3})
        row, _ = self._run(_cp(json.dumps({"answer": "a", "snapshot": {"usage": {"p": 1}, "token_usage": {"u": 3}}})))
        self.assertEqual(row["nano_usage"], {"p": 1})  # usage wins when both are present
        for res in ({"answer": "a"}, {"answer": "a", "snapshot": None}):
            row, _ = self._run(_cp(json.dumps(res)))
            self.assertEqual((row["nano_usage"], row["nano_metrics"], row["nano_stop"]), ({}, None, None))

    def test_unparseable_output_gives_error_row_preferring_stderr_capped_to_tail(self):
        row, _ = self._run(_cp("not json", "E" * 1600 + "END"))
        self.assertEqual(row["answer"], "")
        self.assertEqual(row["wall_s"], 1.23)
        self.assertEqual(row["error"], ("E" * 1600 + "END")[-1500:])
        self.assertNotIn("nano_usage", row)
        row, _ = self._run(_cp("only stdout", ""))
        self.assertEqual(row["error"], "only stdout")
        row, _ = self._run(_cp("", "boom"))  # empty stdout -> IndexError path
        self.assertEqual((row["answer"], row["error"]), ("", "boom"))
        row, _ = self._run(_cp("{\"answer\": \"ok\"}\ngarbage", ""))  # only the LAST line counts
        self.assertIn("error", row)

    def test_crash_with_no_output_still_carries_a_truthy_error(self):
        # main() treats a falsy row["error"] as "no error": a silent nano crash would be scored as a wrong answer
        row, _ = self._run(_cp("", ""))
        self.assertEqual(row["answer"], "")
        self.assertTrue(row["error"])

    def test_valid_json_without_a_dict_answer_gives_error_row_not_a_crash(self):
        # main() only catches TimeoutExpired, so an escaping KeyError/TypeError/AttributeError kills the whole bench run
        for last in ("[1, 2]", "42", '"str"', "null", "{}", '{"answer": null}', '{"answer": 5}'):
            row, _ = self._run(_cp(last, "ERR"))
            self.assertEqual(row, {"answer": "", "wall_s": 1.23, "error": "ERR"}, last)

    def test_non_string_answer_gives_error_row(self):
        # a list survives the [-2000:] slice but later crashes score() (re.findall on a list)
        for last in ('{"answer": ["a"]}', '{"answer": {"k": "v"}}', '{"answer": 5}'):
            row, _ = self._run(_cp(last, "ERR"))
            self.assertEqual(row, {"answer": "", "wall_s": 1.23, "error": "ERR"}, last)

    def test_non_dict_snapshot_is_treated_as_empty_and_keeps_the_answer(self):
        for snap in ("x", [1], 1, True):
            row, _ = self._run(_cp(json.dumps({"answer": "a", "snapshot": snap})))
            self.assertEqual(row, {"answer": "a", "wall_s": 1.23, "nano_usage": {},
                                   "nano_metrics": None, "nano_stop": None}, snap)


if __name__ == "__main__":
    unittest.main()
