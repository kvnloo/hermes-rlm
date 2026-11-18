"""Shutdown while the host answers a sub-call must yield an error reply, not BrokenPipeError."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rlm


class ShutdownSendTests(unittest.TestCase):
    def test_shutdown_during_subcall_returns_error_reply(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = rlm.Session('send', log_dir=Path(tmp.name))
        def serve(req):
            rlm._shutdown()  # kills the kernel before the reply is written
            s.proc.wait(5)
            return {"type": "llm_result", "id": req["id"], "results": ["x"]}
        with mock.patch.object(rlm, '_sessions', {'k': s}), \
             mock.patch.object(rlm, 'sandbox_argv', return_value=[rlm.sys.executable, '-I', str(rlm.KERNEL)]), \
             mock.patch.object(s, '_serve_llm', serve):
            reply = s.exec('print(llm_query("hi"))')
        self.assertIn('kernel exited', reply['error'])
        self.assertIsNone(s.proc)


if __name__ == '__main__':
    unittest.main()
