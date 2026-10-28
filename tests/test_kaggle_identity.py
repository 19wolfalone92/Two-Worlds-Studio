"""Kaggle account discovery without exposing personal API tokens."""
import unittest
from types import SimpleNamespace
from studio.kaggle_identity import parse_username, resolve_username

class KaggleIdentityTests(unittest.TestCase):
    def test_parse_token_backed_config_view(self):
        self.assertEqual(parse_username("- username: example_user\n- auth_method: ACCESS_TOKEN\n"),
                         "example_user")
        self.assertEqual(parse_username("username: studio-account\n"),
                         "studio-account")
    def test_bad_or_missing_config_username(self):
        for value in ("- username: ../../wrong\n", "auth_method: ACCESS_TOKEN\n", ""):
            with self.assertRaises(ValueError):
                parse_username(value)
    def test_explicit_username_does_not_trigger_cli_call(self):
        def fail(*args,**kwargs):
            self.fail("CLI should not run with explicit username")
        self.assertEqual(resolve_username("example_user",runner=fail),"example_user")
        with self.assertRaises(ValueError):
            resolve_username("bad/handle",runner=fail)
    def test_cli_discovery_uses_capture_and_omits_env_override(self):
        def mock(command,**kwargs):
            self.assertEqual(command,["kaggle","config","view"])
            self.assertTrue(kwargs.get("capture_output"))
            self.assertNotIn("KAGGLE_USERNAME",kwargs["env"])
            return SimpleNamespace(stdout="- username: account_2026\n")
        self.assertEqual(resolve_username(runner=mock),"account_2026")

if __name__=="__main__":
    unittest.main()
