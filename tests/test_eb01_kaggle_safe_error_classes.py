"""Off-line security tests: never echo private Kaggle stdout, preserve API error class."""
import unittest
from tools.eb01_true_gpu_clip_numbers import error_family_only

class ErrorSafeKaggleReadTests(unittest.TestCase):
    def test_429_throttle_requires_backoff_no_blind_loop(self):
        self.assertEqual(error_family_only("HTTP 429 Too Many Requests\nPRIVATE_TOKEN=sensitive"),
                         "HTTP_429_KAGGLE_RATE_LIMIT_PAUSE_REQUESTS")
    def test_token_auth_and_dataset_acl_are_different(self):
        self.assertEqual(error_family_only("HTTP 401 Unauthorized"),
                         "KAGGLE_TOKEN_UNAUTHORIZED")
        self.assertEqual(error_family_only("HTTP 403 Forbidden"),
                         "KAGGLE_DATASET_OR_KERNEL_ACL_DENIED")
    def test_private_kernel_missing_not_same_as_rate_limit(self):
        self.assertEqual(error_family_only("API: 404 not found"),
                         "KAGGLE_PRIVATE_KERNEL_OR_OUTPUT_NOT_FOUND")
    def test_server_and_network(self):
        self.assertEqual(error_family_only("503 Service Unavailable"),
                         "KAGGLE_API_SERVER_UNAVAILABLE")
        self.assertEqual(error_family_only("SSL certificate error"),
                         "KAGGLE_NETWORK_OR_TLS_FAILURE")
    def test_no_raw_credentials_leaked_from_unknown_error(self):
        self.assertEqual(error_family_only("X-PASSWORD=my-credential"),
                         "KAGGLE_NONZERO_EXIT_UNCLASSIFIED")

if __name__=="__main__":
    unittest.main()
