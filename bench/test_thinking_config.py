"""Generated child config leaves the explicit thinking policy to the proxy."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import paper_matrix as matrix
from paper_proxy import Ledger, ROOT, SUB, Router


class ThinkingConfigTests(unittest.TestCase):
    def test_config_and_proxy_respect_explicit_flag(self):
        try:
            import yaml
        except ImportError:
            if os.environ.get('BENCH_REQUIRE_DEPS') == '1':
                raise
            self.skipTest('PyYAML is required for generated-home checks')
        for disabled in (False, True):
            with self.subTest(disable_thinking=disabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                template, plugin = root / 'template', root / 'plugin'
                template.mkdir()
                plugin.mkdir()
                (plugin / '__init__.py').write_text('')
                args = SimpleNamespace(max_subcalls=2, sub_max_tokens=64, cell_timeout=10,
                    max_context_chars=1000, root_model=ROOT, sub_model=SUB,
                    max_tokens=64, max_turns=4, disable_thinking=disabled)
                config = matrix.fresh_home(root / 'home', template, plugin, 'released-rlm',
                    SimpleNamespace(url='http://unused.invalid'), args)
                written = yaml.safe_load((root / 'home' / 'config.yaml').read_text())
                self.assertEqual(config, written)
                auxiliary = written['auxiliary']['rlm_subcall']
                # A generated override would reach the proxy even in default mode.
                extra = auxiliary.get('extra_body', {})
                bodies = []
                class Response(io.BytesIO):
                    status = 200
                    headers = {'Content-Type': 'application/json'}
                def upstream(request, **kwargs):
                    bodies.append(json.loads(request.data))
                    return Response(b'{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}')
                with Router(Ledger(root / 'ledger.db'), 'policy', disable_thinking=disabled) as router, \
                        patch.dict(os.environ, NVIDIA_API_KEY='offline-fixture'), \
                        patch('urllib.request.urlopen', side_effect=upstream) as request:
                    router.forward({'messages': []}, ROOT, 'root')
                    router.forward({'messages': [], **extra}, SUB, 'sub')
                    self.assertEqual(request.call_count, 2)
                for body in bodies:
                    if disabled:
                        self.assertIs(body['chat_template_kwargs']['enable_thinking'], False)
                    else:
                        self.assertNotIn('enable_thinking', body.get('chat_template_kwargs', {}))
