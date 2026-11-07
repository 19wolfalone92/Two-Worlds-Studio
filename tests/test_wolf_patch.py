import tempfile
import unittest
import io
import json
from unittest.mock import patch
from pathlib import Path
from wolf_ai.patch_proposal import select_files, validate_patch, propose


class PatchProposalTests(unittest.TestCase):
    def test_selection_rejects_outside_and_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'hello.py').write_text('x = 1\n')
            self.assertEqual(select_files(root, ['hello.py']), {'hello.py': 'x = 1\n'})
            for name in ('../hello.py', '.env', 'file.txt', '/tmp/hello.py'):
                with self.assertRaises(ValueError):
                    select_files(root, [name])

    def test_patch_limits_to_selected_files(self):
        safe = 'diff --git a/hello.py b/hello.py\n--- a/hello.py\n+++ b/hello.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n'
        validate_patch(safe, {'hello.py'})
        with self.assertRaises(ValueError):
            validate_patch(safe.replace('b/hello.py', 'b/.env'), {'hello.py'})
        with self.assertRaises(ValueError):
            validate_patch(safe.replace('diff --git', 'new file mode 100644\ndiff --git'), {'hello.py'})

    def test_no_source_disclosure_without_opt_in(self):
        with self.assertRaises(ValueError):
            propose(Path('.'), [], 'Fix broken output', base_url='https://example.com/v1',
                    model='x', output=Path('/tmp/p.patch'), send_source=False)

    def test_mocked_llm_proposes_applicable_patch_without_changing_source(self):
        patch_text = ('diff --git a/hello.py b/hello.py\n'
                      '--- a/hello.py\n+++ b/hello.py\n'
                      '@@ -1 +1 @@\n-x = 1\n+x = 2\n')
        response = {'choices': [{'message': {'content': patch_text}}]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'src'
            source.mkdir()
            original = source / 'hello.py'
            original.write_text('x = 1\n')
            destination = root / 'change.patch'
            with patch('wolf_ai.patch_proposal.urlopen',
                       return_value=io.BytesIO(json.dumps(response).encode())):
                result = propose(source, ['hello.py'], 'Change constant to two',
                                 base_url='https://example.com/v1', model='fake',
                                 output=destination, send_source=True)
            self.assertEqual(result['git_apply_check'], 'passed')
            self.assertEqual(original.read_text(), 'x = 1\n')
            self.assertEqual(destination.read_text(), patch_text)
