"""Exercise real Hermes discovery/unload with synthetic, non-spawning sessions.

RLM_HERMES_SOURCE=/path/to/hermes python -m unittest discover -s tests
Use an interpreter with that host's dependencies installed. No model calls.
"""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class HostUnloadTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('RLM_HERMES_SOURCE'), 'requires real Hermes checkout')
    def test_unload_disposes_sessions_once_and_clears_registry(self):
        root = Path(__file__).resolve().parents[1]
        script = textwrap.dedent('''\
            import json, os, shutil, sys, tempfile
            from pathlib import Path
            from types import SimpleNamespace
            sys.path.insert(0, os.environ['RLM_HERMES_SOURCE'])
            with tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                os.environ['HERMES_HOME'] = str(home)
                (home / 'config.yaml').write_text(json.dumps({
                    'plugins': {'enabled': ['rlm']}, 'telemetry': {'enabled': False}}))
                shutil.copytree(sys.argv[1], home / 'plugins' / 'rlm',
                                ignore=shutil.ignore_patterns('.git', '__pycache__', 'tests'))
                from hermes_cli.plugins import get_plugin_manager
                manager = get_plugin_manager()
                manager.discover_and_load()
                loaded = manager._plugins['rlm']
                assert loaded.enabled and not loaded.error, loaded.error
                implementation = sys.modules[loaded.module.__name__ + '.rlm']
                killed = []
                def kill():
                    # Cleanup must detach state under the lock but kill outside it.
                    assert implementation._sessions_lock.acquire(blocking=False)
                    implementation._sessions_lock.release()
                    killed.append('kill')
                def fail_kill():
                    raise OSError('synthetic kill failure')
                implementation._sessions['failing'] = SimpleNamespace(kill=fail_kill)
                implementation._sessions['synthetic'] = SimpleNamespace(kill=kill)
                try:
                    assert manager.unload('rlm')
                    assert killed == ['kill'], tuple(killed)
                    assert implementation._sessions == {}, implementation._sessions
                    implementation._shutdown()  # process-exit cleanup is idempotent
                    assert killed == ['kill'], tuple(killed)
                finally:
                    implementation._shutdown()
        ''')
        result = subprocess.run([sys.executable, '-c', script, str(root)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
