"""Pin register(): tool->schema/handler/toolset mapping, aux-task name used by default_subcall, command wiring, _CTX."""
import re
import unittest
from pathlib import Path
from unittest import mock
import rlm


class Ctx:
    def __init__(self):
        self.tools, self.aux, self.cmds, self.completes = {}, {}, {}, []
        self.llm = self

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = (toolset, schema, handler, kw)

    def register_auxiliary_task(self, name, **kw):
        self.aux[name] = kw

    def register_command(self, name, handler, **kw):
        self.cmds[name] = (handler, kw)

    def complete(self, **kw):
        self.completes.append(kw)
        u = mock.Mock(input_tokens=1, output_tokens=2, cost_usd=0.0)
        return mock.Mock(text="t", usage=u)


class RegisterPins(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(rlm, '_CTX', None)
        p.start()
        self.addCleanup(p.stop)
        self.ctx = Ctx()
        rlm.register(self.ctx)

    def test_tool_wiring(self):
        self.assertEqual(set(self.ctx.tools), {'rlm_load', 'rlm_exec'})
        for name, schema, handler in (('rlm_load', rlm.LOAD_SCHEMA, rlm._load_handler),
                                      ('rlm_exec', rlm.EXEC_SCHEMA, rlm._exec_handler)):
            toolset, s, h, kw = self.ctx.tools[name]
            self.assertEqual(toolset, rlm.TOOLSET)
            self.assertIs(s, schema)
            self.assertIs(h, handler)
            self.assertEqual(kw['description'], schema['description'])
            self.assertEqual(kw['emoji'], '🧮')

    def test_manifest_provides_tools_match(self):
        text = (Path(rlm.__file__).parent / 'plugin.yaml').read_text()
        listed = re.findall(r'(?m)^  - (\S+)$', text.split('provides_tools:')[1])
        self.assertEqual(sorted(listed), sorted(self.ctx.tools))

    def test_aux_task_is_the_one_default_subcall_uses(self):
        self.assertEqual(list(self.ctx.aux), ['rlm_subcall'])
        rlm.default_subcall('p', 7)
        self.assertEqual(self.ctx.completes[0]['task'], next(iter(self.ctx.aux)))

    def test_command_and_ctx(self):
        self.assertIs(rlm._CTX, self.ctx)
        handler, kw = self.ctx.cmds['rlm']
        self.assertEqual(list(self.ctx.cmds), ['rlm'])
        self.assertIs(handler, rlm._rlm_command)
        self.assertTrue(kw['description'])


if __name__ == '__main__':
    unittest.main()
