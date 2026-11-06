"""Entirely offline tests for free provider planning and one-shot FLUX image adapter."""
from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib import error

from PIL import Image

from studio.cloudflare_flux import (
    MODEL, FreeImageError, checked_reference, multipart,
    parse_image_response, plan, render_one, request_image,
)
from studio.free_studio import readiness, scene_route, prepare_batch


def png(width=256, height=320):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (42, 56, 70)).save(buffer, format="PNG")
    return buffer.getvalue()


def shot(scene="S01"):
    return {"scene":scene, "prompt":"Cinematic abandoned corridor in heavy rain, realistic light",
            "seed":12345, "references":{}}


class FakeResponse:
    def __init__(self, response: bytes):
        self.response = response
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, n):
        return self.response[:n]


class CloudflareFluxTests(unittest.TestCase):
    def test_dry_run_no_auth_or_images(self):
        report = plan(shot(), 256, 320)
        self.assertFalse(report["image_generated"])
        self.assertEqual(report["review_status"], "unreviewed")
        self.assertFalse(report["ready_for_publication"])
        self.assertNotIn("prompt", json.dumps(report))

    def test_reference_size_and_multipart(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"ref.png"
            path.write_bytes(png(300,400))
            payload, mimetype, fingerprints = multipart(shot()["prompt"],256,320,12345,[path])
            self.assertIn("multipart/form-data", mimetype)
            self.assertIn(b'name="input_image_0"',payload)
            self.assertNotIn(b"__REFERENCE_IMAGE__",payload)
            self.assertEqual(len(fingerprints),1)
            self.assertIn(b'name="seed"',payload)
            path.write_bytes(png(512,512))
            with self.assertRaisesRegex(ValueError, "smaller than"):
                checked_reference(path)

    def test_references_max_four_and_dimension_validation(self):
        with self.assertRaisesRegex(ValueError,"four"):
            multipart(shot()["prompt"],256,320,12345,[Path("missing.png")]*5)
        with self.assertRaises(ValueError):
            plan(shot(),100,320)
        with self.assertRaises(ValueError):
            plan({**shot(),"seed":True},256,320)

    def test_real_image_response_json_and_pixel_validation(self):
        original=png()
        payload=json.dumps({"success":True,"result":{"image":base64.b64encode(original).decode()}}).encode()
        decoded=parse_image_response(payload,256,320)
        with Image.open(io.BytesIO(decoded)) as actual:
            self.assertEqual(actual.size,(256,320))
            self.assertEqual(actual.format,"PNG")
        with self.assertRaises(FreeImageError):
            parse_image_response(payload,256,256)
        with self.assertRaises(FreeImageError):
            parse_image_response(b'{"success":false}',256,320)
        with self.assertRaises(FreeImageError):
            parse_image_response(b'{"result":{"image":"not base64!"}}',256,320)
        with self.assertRaises(FreeImageError):
            parse_image_response(b'{"result":{"image":"'+base64.b64encode(b"not image")+b'"}}',256,320)

    def test_http_request_fixed_host_no_secret_in_results(self):
        secret="cf-secret-example"
        calls=[]
        response=json.dumps({"success":True,"result":{"image":base64.b64encode(png()).decode()}}).encode()
        def opener(request, timeout):
            calls.append(request)
            self.assertEqual(timeout,100)
            return FakeResponse(response)
        raw,shas=request_image(token=secret,account_id="e"*32,
                        prompt=shot()["prompt"],width=256,height=320,
                        seed=12345,references=[],opener=opener)
        self.assertEqual(shas,[])
        self.assertEqual(calls[0].get_method(),"POST")
        self.assertEqual(calls[0].get_header("Authorization"),"Bearer "+secret)
        self.assertTrue(calls[0].full_url.startswith("https://api.cloudflare.com/client/v4/accounts/"))
        self.assertIn(MODEL,calls[0].full_url)
        self.assertEqual(Image.open(io.BytesIO(raw)).size,(256,320))

    def test_http_429_fails_without_leaking_provider_body(self):
        secret="secret_do_not_print"
        def opener(req,timeout):
            raise error.HTTPError(req.full_url,429,secret,{},io.BytesIO(secret.encode()))
        with self.assertRaises(FreeImageError) as e:
            request_image(token=secret,account_id="e"*32,
                          prompt=shot()["prompt"],width=256,height=320,
                          seed=12345,references=[],opener=opener)
        self.assertNotIn(secret,str(e.exception))
        self.assertIn("429",str(e.exception))

    def test_one_shot_receipt_and_no_overwrite_or_network_on_existing_target(self):
        with tempfile.TemporaryDirectory() as td:
            calls=[]
            def client(**kwargs):
                calls.append(kwargs)
                return png(),[]
            result=render_one(shot(),Path(td),token="mock",account_id="e"*32,
                              width=256,height=320,client=client)
            self.assertEqual(len(calls),1)
            self.assertTrue((Path(td)/"S01.png").is_file())
            self.assertEqual(result["review_status"],"unreviewed")
            self.assertFalse(result["master_ready"])
            self.assertFalse(result["published"])
            self.assertFalse(result["ready_for_publication"])
            self.assertTrue((Path(td)/"S01.art_review.json").is_file())
            with self.assertRaises(FileExistsError):
                render_one(shot(),Path(td),token="mock",account_id="e"*32,
                           width=256,height=320,client=client)
            self.assertEqual(len(calls),1)

    def test_public_repo_output_path_rejected_before_network(self):
        import studio.cloudflare_flux as cf
        with self.assertRaisesRegex(ValueError,"public source repository"):
            render_one(shot(),cf.ROOT/"dist"/"private-art",
                       token="x",account_id="e"*32,width=256,height=320,
                       client=lambda **kw: (_ for _ in ()).throw(AssertionError("network was used")))


class DispatcherTests(unittest.TestCase):
    def test_connected_not_authenticated_or_media_ready(self):
        env={"CLOUDFLARE_AI_TOKEN":"dummy","CLOUDFLARE_ACCOUNT_ID":"e"*32,
             "GEMINI_API_KEY":"dummy","OPENROUTER_API_KEY":"dummy",
             "HORDE_API_KEY":"dummy"}
        report=readiness(env)
        self.assertTrue(report["services"]["cloudflare_image"]["configured"])
        self.assertIsNone(report["services"]["cloudflare_image"]["authenticated"])
        self.assertFalse(report["services"]["cloudflare_image"]["inference_verified"])
        self.assertFalse(report["services"]["groq_optional"]["configured"])
        self.assertFalse(report["automated_publication"])

    def test_dry_run_routes_and_privacy(self):
        env={"CLOUDFLARE_AI_TOKEN":"fake","CLOUDFLARE_ACCOUNT_ID":"e"*32,
             "HORDE_API_KEY":"fake"}
        pack={"schema":1,"episode":"E01",
              "scenes":[shot("S01"),shot("S02")]}
        report=prepare_batch(pack,env)
        self.assertEqual(report["scene_count"],2)
        self.assertEqual(report["routes"][0]["provider"],"cloudflare_image")
        self.assertTrue(report["routes"][0]["private"])
        self.assertFalse(report["auto_publish"])
        self.assertFalse(report["script_modified"])
        self.assertNotIn("fake",json.dumps(report))

    def test_horde_only_for_public_generic_without_refs(self):
        env={"HORDE_API_KEY":"fake"}
        status=readiness(env)["services"]
        self.assertIsNone(scene_route(shot(),status)["provider"])
        self.assertEqual(scene_route(shot(),status,public_generic=True)["provider"],"horde")
        with_ref={**shot(),"references":{"person":{"file":"private.png"}}}
        self.assertIsNone(scene_route(with_ref,status,public_generic=True)["provider"])

    def test_untrusted_duplicate_job_and_bad_prompt_fail(self):
        pack={"schema":1,"scenes":[shot("S01"),shot("S01")]}
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            prepare_batch(pack,{})
        with self.assertRaises(ValueError):
            prepare_batch({"schema":1,"scenes":[{"scene":"S01","prompt":"hi","seed":1}]},{})


if __name__ == "__main__":
    unittest.main()
