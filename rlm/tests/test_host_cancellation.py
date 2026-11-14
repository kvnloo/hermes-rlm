"""Opt-in real Hermes transport cancellation against a hanging local HTTP server.

reuse-checked: tools.interrupt.set_interrupt and PluginManager discovery, no fake LLM.
"""
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get('RLM_HERMES_SOURCE'),
                    reason='set RLM_HERMES_SOURCE to opt in to actual Hermes integration')
def test_host_stop_closes_inflight_http_connection(tmp_path, monkeypatch, request):
    home = tmp_path / 'hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('RLM_MAX_SUBCALLS', '99')
    monkeypatch.syspath_prepend(os.environ['RLM_HERMES_SOURCE'])
    started, disconnected = threading.Event(), threading.Event()

    class HangingHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            started.set()
            self.connection.settimeout(4)
            if self.rfile.read(1) == b'':
                disconnected.set()

    server = HTTPServer(('127.0.0.1', 0), HangingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f'http://127.0.0.1:{server.server_port}/v1'
    config = {
        'model': {'provider': 'local-test', 'default': 'test-open-weight'},
        'plugins': {'enabled': ['rlm'], 'entries': {'rlm': {'settings': {
            'concurrency': 1, 'max_subcalls': 3, 'timeout_s': 10}}}},
        # Plain HTTP loopback has no certificate to validate. Avoid cold trust-store IO.
        'providers': {'local-test': {'base_url': url, 'api_key': 'local-test-placeholder',
                                     'ssl_verify': False}},
        'auxiliary': {'transient_retries': 0, 'rlm_subcall': {
            'provider': 'local-test', 'model': 'test-open-weight', 'base_url': url,
            'api_key': 'local-test-placeholder', 'fallback_models': []}},
        'telemetry': {'enabled': False},
    }
    # JSON is valid YAML; no PyYAML dependency for the portable test suite.
    (home / 'config.yaml').write_text(json.dumps(config))
    shutil.copytree(Path(__file__).parents[1], home / 'plugins' / 'rlm',
                    ignore=shutil.ignore_patterns('tests', '__pycache__'))
    from hermes_cli.plugins import get_plugin_manager
    from tools.interrupt import set_interrupt
    __import__('agent.auxiliary_client')
    __import__('openai.resources.chat')

    manager = get_plugin_manager()
    manager.discover_and_load()
    loaded = manager._plugins['rlm']
    assert loaded.enabled and not loaded.error, loaded.error
    plugin = loaded.module
    assert plugin._CTX.get_config('max_subcalls') == 3
    assert plugin.limits()['max_subcalls'] == 3
    s = plugin.Session('http-cancel', allow_unsandboxed=not request.config.getoption('--rlm-sandbox'))
    result = {}
    worker = threading.Thread(target=lambda: result.update(s.exec("llm_batch(['first','queued'])")))
    try:
        worker.start()
        if not started.wait(5):
            import sys
            import traceback
            frame = sys._current_frames().get(worker.ident)
            if frame is not None:
                traceback.print_stack(frame)
            pytest.fail(f'actual host lane did not dispatch; worker result: {result}')
        t0 = time.monotonic()
        set_interrupt(True, thread_id=worker.ident)
        worker.join(2)
        assert not worker.is_alive(), 'host stop must not wait for request timeout'
        assert time.monotonic() - t0 < 2
        assert disconnected.wait(2), 'actual HTTP request connection must close on cancel'
        assert 'cancel' in result['error'].lower()
        assert s.stats['subcalls'] == 1
        assert s.proc is None
        print('HTTP peer observed EOF; host stop cancelled 1 attempt and skipped queued prompt')
    finally:
        set_interrupt(False, thread_id=worker.ident)
        s.kill()
        worker.join(5)
        manager.unload()
        server.shutdown()
        server.server_close()
        server_thread.join(2)
