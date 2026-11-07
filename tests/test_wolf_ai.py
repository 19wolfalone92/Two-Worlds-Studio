"""Offline contract tests for the bounded orchestrator. No external services."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wolf_ai.ai_review import suggest
from wolf_ai.orchestrator import PROJECTS, clean_log, recipes, run


class WolfTests(unittest.TestCase):
    def test_all_projects_have_fixed_recipes(self):
        for name in PROJECTS:
            self.assertTrue(recipes(name, 'audit'))
            self.assertTrue(recipes(name, 'test'))
        with self.assertRaises(ValueError):
            recipes('../private', 'audit')
        with self.assertRaises(ValueError):
            recipes('studio', 'trade')

    def test_planning_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            source = base / 'src'
            source.mkdir()
            report = base / 'reports' / 'run.json'
            actual = run('studio', 'test', source, report)
            self.assertEqual(actual['status'], 'planned')
            self.assertFalse(actual['execute'])
            self.assertTrue(report.is_file())
            self.assertEqual(json.loads(report.read_text())['project'], 'studio')

    def test_never_write_report_into_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                run('studio', 'audit', root, root / 'audit.json')
            with self.assertRaises(ValueError):
                run('studio', 'audit', root, root.parent / 'x.json', budget_seconds=10000)

    def test_bounded_real_execution_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'src').mkdir()
            with patch('wolf_ai.orchestrator.recipes', return_value=()):
                actual = run('studio', 'audit', root / 'src', root / 'out.json', execute=True)
            self.assertEqual(actual['status'], 'passed')

    def test_secrets_redacted_and_trimmed(self):
        s = clean_log('Authorization: bearer abc123\napi_key=SECRET_TOKEN\n')
        self.assertNotIn('abc123', s)
        self.assertNotIn('SECRET_TOKEN', s)
        self.assertTrue(clean_log('x' * 17000).endswith('[TRUNCATED]'))

    def test_ai_review_off_without_explicit_opt_in(self):
        self.assertEqual(suggest({'schema': 1}, base_url='https://example.com/v1', model='x')['status'], 'disabled')
        with self.assertRaises(ValueError):
            suggest({'schema': 1}, base_url='http://example.com/v1', model='x', send=True)


if __name__ == '__main__':
    unittest.main()
