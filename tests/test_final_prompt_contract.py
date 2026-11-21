"""The registered model-facing FINAL example passes the actual span contract."""
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

import rlm


class FinalPromptContractTests(unittest.TestCase):
    def test_advertised_final_repairs_uncited_rejection_and_names_followup(self):
        ctx = Mock()
        ctx.get_config.side_effect = lambda key, default: default
        with patch.object(rlm, "_CTX", None):
            rlm.register(ctx)
        tool = next(call for call in ctx.register_tool.call_args_list if call.args[0] == "rlm_exec")
        description = tool.kwargs["description"]
        self.assertIs(tool.args[2], rlm.EXEC_SCHEMA)
        # Parse and check the advertised source span argument, without executing examples.
        example = re.search(r"`(FINAL\(answer[^\n`]*\))`", description).group(1)
        self.assertEqual(example, "FINAL(answer, citations=[cite(source, start, end)])")
        with tempfile.TemporaryDirectory() as tmp:
            session = rlm.Session("prompt-fixture", log_dir=Path(tmp))
            session.sources = {"context.txt": "Answer B"}
            with self.assertRaisesRegex(ValueError, "citation required"):
                session._validate_final("B", [])
            self.assertEqual(session._validate_final("B", [
                {"source": "context.txt", "start": 7, "end": 8},
            ])[0]["quote"], "B")
        self.assertIn("Check that rlm_exec returns a nonempty final and error=null", description)
        self.assertIn("another cell clears the current validated final", description)
