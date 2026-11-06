"""Offline read-only Cloudflare account/model preflight tests."""
from __future__ import annotations
import io
import json
import unittest
from urllib import error
from studio.free_connectivity import check_cloudflare

class Response:
    def __init__(self,obj):
        self.body=json.dumps(obj).encode()
    def __enter__(self): return self
    def __exit__(self,*args):return False
    def read(self,n): return self.body[:n]

class ConnectionTests(unittest.TestCase):
    def test_token_and_account_are_not_echoed(self):
        urls=[]
        def opener(req,timeout):
            urls.append(req.full_url)
            self.assertEqual(req.get_method(),"GET")
            self.assertEqual(timeout,20)
            self.assertEqual(req.get_header("Authorization"),"Bearer private_token")
            return Response({"success":True,"result":[{"name":"@cf/black-forest-labs/flux-2-klein-4b"}]})
        report=check_cloudflare("private_token","e"*32,opener=opener)
        self.assertTrue(report["ready"])
        self.assertFalse(report["generated"])
        self.assertFalse(report["quota_checked"])
        self.assertNotIn("private_token",json.dumps(report))
        self.assertNotIn("e"*32,json.dumps(report))
        self.assertEqual(len(urls),1)
        self.assertIn("/ai/models/search?",urls[0])

    def test_missing_account_is_non_credential_failure(self):
        self.assertEqual(check_cloudflare("token","invalid")["status"],
                         "missing_or_invalid_account_id")
        self.assertEqual(check_cloudflare("","e"*32)["status"],"missing_token")

    def test_403_and_unlisted_model_not_treated_as_success(self):
        def bad(req,timeout):
            raise error.HTTPError(req.full_url,403,"secret_error",{},io.BytesIO(b"secret"))
        report=check_cloudflare("token","e"*32,opener=bad)
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"],"http_403")
        self.assertNotIn("secret",json.dumps(report))
        def other(req,timeout):
            return Response({"success":True,"result":[{"name":"other-model"}]})
        report=check_cloudflare("token","e"*32,opener=other)
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"],"auth_ok_model_unconfirmed")

if __name__=="__main__":
    unittest.main()
