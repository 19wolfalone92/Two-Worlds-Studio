"""Read-only Cloudflare account/model readiness; no GPU or inference credits."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from studio.cloudflare_flux import ACCOUNT_ID, BASE, MODEL


def check_cloudflare(token: str, account_id: str, *, opener=urllib.request.urlopen) -> dict:
    if not token:
        return {"ready": False, "status": "missing_token", "generated": False}
    if not ACCOUNT_ID.fullmatch(account_id or ""):
        return {"ready": False, "status": "missing_or_invalid_account_id", "generated": False}
    url = BASE + account_id + "/ai/models/search?" + urllib.parse.urlencode({
        "search": "flux-2-klein-4b", "per_page": "20"})
    req = urllib.request.Request(url,headers={"Authorization": "Bearer " + token,
                                              "Accept": "application/json"},method="GET")
    try:
        with opener(req,timeout=20) as response:
            raw = response.read(2*1024*1024+1)
    except urllib.error.HTTPError as exc:
        return {"ready": False, "status": "http_"+str(exc.code), "generated": False}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"ready": False, "status": "network_or_timeout", "generated": False}
    if len(raw)>2*1024*1024:
        return {"ready": False, "status": "response_too_large", "generated": False}
    try:
        payload=json.loads(raw)
    except (ValueError,UnicodeError):
        return {"ready": False, "status": "invalid_json", "generated": False}
    if not isinstance(payload,dict) or payload.get("success") is not True:
        return {"ready": False, "status": "api_response_not_successful", "generated": False}
    models=payload.get("result")
    if not isinstance(models,list):
        return {"ready": False, "status": "unknown_model_list_shape", "generated": False}
    def matches(item):
        if not isinstance(item,dict):
            return False
        return any(MODEL in str(item.get(key,"")) or "flux-2-klein-4b" in str(item.get(key,"")).lower()
                   for key in ("name","id","model","model_id"))
    found=any(matches(entry) for entry in models)
    return {"ready": found, "status": "model_visible" if found else "auth_ok_model_unconfirmed",
            "model": MODEL, "generated": False, "quota_checked": False,
            "note": "Read-only model listing cannot prove paid/free plan, current daily quota or image inference."}


def main():
    p=argparse.ArgumentParser(description="Safe read-only Cloudflare Workers AI account/model probe")
    p.add_argument("--check",action="store_true",required=True)
    args=p.parse_args()
    result=check_cloudflare(os.environ.get("CLOUDFLARE_AI_TOKEN",""),
                            os.environ.get("CLOUDFLARE_ACCOUNT_ID",""))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
