import unittest
from types import SimpleNamespace
from tools.eb01_kaggle2_auth_probe import check,config_name

class AccountAuthSafetyTest(unittest.TestCase):
    def test_two_different_verified_usernames(self):
        def runner(argv,env,capture_output,text,timeout):
            name="primary_user" if env["KAGGLE_API_TOKEN"]=="owner-token" else "brother_user"
            if argv[1:3]==["config","view"]:
                return SimpleNamespace(returncode=0,stdout="username: "+name,stderr="")
            self.assertEqual(argv[:3],["kaggle","kernels","list"])
            self.assertIn("--mine",argv)
            return SimpleNamespace(returncode=0,stdout="PRIVATE_DATA_NOT_EXPOSED",stderr="")
        result=check("owner-token","brother-token",runner)
        self.assertTrue(result["two_distinct_authenticated_accounts"])
        self.assertFalse(result["originals_copied"])
        self.assertFalse(result["gpu_inference_launched"])
        self.assertFalse(result["brother_gpu_connected"])
    def test_absent_brother_key_blocks_connection(self):
        def runner(argv,env,capture_output,text,timeout):
            return SimpleNamespace(returncode=0,stdout="username: owner",stderr="")
        result=check("owner","",runner)
        self.assertEqual(result["brother_account"],"MISSING_GITHUB_SECRET")
        self.assertFalse(result["two_distinct_authenticated_accounts"])
    def test_second_token_same_account_is_not_separate_compute(self):
        def runner(argv,env,capture_output,text,timeout):
            return SimpleNamespace(returncode=0,stdout="username: owner",stderr="")
        result=check("owner","different-token-same-account",runner)
        self.assertFalse(result["two_distinct_authenticated_accounts"])
    def test_cli_auth_failure_not_authenticated(self):
        def runner(argv,env,capture_output,text,timeout):
            return SimpleNamespace(returncode=1,stdout="",stderr="secret-would-be-here")
        result=check("owner","brother",runner)
        self.assertEqual(result["original_account"],"TOKEN_FORMAT_OR_CLI_REJECTED")
        self.assertEqual(result["brother_account"],"TOKEN_FORMAT_OR_CLI_REJECTED")
        self.assertNotIn("secret-would-be-here",str(result))
    def test_bad_username_never_returned(self):
        self.assertIsNone(config_name("username: ../../secrets"))
        self.assertEqual(config_name("username: Owner_2026"),"owner_2026")

if __name__=="__main__":
    unittest.main()
