"""Pin Session._serve_llm worker pool: concurrency is floored at 1 and really bounds parallelism."""
import tempfile
import threading
import unittest
from pathlib import Path
import rlm


class ConcurrencyPins(unittest.TestCase):
    def session(self, concurrency, subcall):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = rlm.Session('t', subcall=subcall, log_dir=Path(tmp.name))
        s.lim = {**s.lim, 'concurrency': concurrency}
        return s

    def serve(self, s, n):
        return s._serve_llm({'id': 1, 'prompts': [f'p{i}' for i in range(n)]})

    def peak_tracker(self):
        busy, peak, lock = [0], [0], threading.Lock()

        def track(p, m):
            with lock:
                busy[0] += 1
                peak[0] = max(peak[0], busy[0])
            threading.Event().wait(0.05)
            with lock:
                busy[0] -= 1
            return p.upper(), {}

        return track, peak

    def test_concurrency_one_and_nonpositive_serialize(self):
        for c in (1, 0, -3):
            with self.subTest(concurrency=c):
                track, peak = self.peak_tracker()
                self.assertEqual(self.serve(self.session(c, track), 3)['results'], ['P0', 'P1', 'P2'])
                self.assertEqual(peak[0], 1)

    def test_concurrency_two_overlaps(self):
        barrier = threading.Barrier(2, timeout=2)

        def meet(p, m):
            barrier.wait()
            return 'ok', {}

        self.assertEqual(self.serve(self.session(2, meet), 2)['results'], ['ok', 'ok'])


if __name__ == "__main__":
    unittest.main()
