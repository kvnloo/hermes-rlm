"""Pin default_subcall (host LLM lane call + usage mapping) and the atexit _shutdown kill sweep."""
import unittest
from types import SimpleNamespace as NS
from unittest import mock
import rlm


class DefaultSubcallPins(unittest.TestCase):
    def ctx(self, text="hi", cost=0.5):
        usage = NS(input_tokens=7, output_tokens=3, cost_usd=cost)
        ctx = mock.Mock()
        ctx.llm.complete.return_value = NS(text=text, usage=usage)
        return ctx

    def test_unregistered_raises(self):
        with mock.patch.object(rlm, "_CTX", None):
            with self.assertRaisesRegex(RuntimeError, "not registered"):
                rlm.default_subcall("p", 10)

    def test_calls_host_lane_with_user_message_and_labels(self):
        ctx = self.ctx()
        with mock.patch.object(rlm, "_CTX", ctx):
            rlm.default_subcall("the prompt", 123)
        ctx.llm.complete.assert_called_once_with(
            messages=[{"role": "user", "content": "the prompt"}], max_tokens=123,
            task="rlm_subcall", purpose="rlm.subcall")

    def test_returns_text_and_mapped_usage(self):
        with mock.patch.object(rlm, "_CTX", self.ctx()):
            self.assertEqual(rlm.default_subcall("p", 1),
                             ("hi", {"prompt_tokens": 7, "completion_tokens": 3, "cost_usd": 0.5}))

    def test_none_text_becomes_empty_string(self):
        with mock.patch.object(rlm, "_CTX", self.ctx(text=None)):
            self.assertEqual(rlm.default_subcall("p", 1)[0], "")


class ShutdownPins(unittest.TestCase):
    def test_kills_every_session(self):
        a, b = mock.Mock(), mock.Mock()
        with mock.patch.dict(rlm._sessions, {"a": a, "b": b}, clear=True):
            rlm._shutdown()
        a.kill.assert_called_once_with()
        b.kill.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
