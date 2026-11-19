"""Kaggle v2 token diagnostic for TWO AUTHORIZED accounts, CPU/read-only.
Never print/log tokens, usernames, private datasets, kernel names or raw stderr.
One authenticated read-only private-kernel-list attempt per configured secret.
"""
from __future__ import annotations
import json,os,subprocess,sys
from pathlib import Path

def shape_and_env():
    raw=os.environ.get("KAGGLE_API_TOKEN","").strip()
    if not raw:return "missing_secret",None
    env=os.environ.copy()
    # Some legacy stored secrets are the entire kaggle.json rather than KGAT_ tokens.
    # Kaggle CLI only supports username+key in that format; do not expose either.
    if raw.startswith("{"):
        try:
            doc=json.loads(raw)
            username,key=doc.get("username"),doc.get("key")
            if not isinstance(username,str) or not isinstance(key,str) or not username or not key:
                return "legacy_json_invalid",None
            env.pop("KAGGLE_API_TOKEN",None)
            env["KAGGLE_USERNAME"]=username
            env["KAGGLE_KEY"]=key
            return "legacy_username_key",env
        except (ValueError,TypeError):
            return "legacy_json_invalid",None
    return "personal_access_token",env

def classify(s):
    s=s.lower()
    if "429" in s or "too many requests" in s or "rate limit" in s:return "rate_limited_429"
    if "401" in s or "unauthorized" in s or "authentication required" in s:return "authentication_401"
    if "403" in s or "forbidden" in s:return "forbidden_403"
    if "expired" in s:return "expired_token"
    if "invalid" in s or "token" in s or "credential" in s or "authenticate" in s or "login" in s:return "authentication_or_token_invalid"
    if "not enough values to unpack" in s:return "upstream_cli_duration_bug"
    if "unexpected argument" in s or "unrecognized arguments" in s:return "kaggle_cli_argument_error"
    if "not found" in s or "404" in s:return "resource_404"
    if "ssl" in s or "dns" in s or "name resolution" in s:return "network_unavailable"
    return "unclassified_cli_error"

def run():
    mode,env=shape_and_env()
    report={"type":"T-Flex/Kaggle-authorization-readonly","credential_format":mode,
            "account_verified":False,"gpu_started":False,"images_transferred":False,
            "credentials_printed":False,"result":"not_attempted"}
    if env is not None:
        try:
            p=subprocess.run(["kaggle","kernels","list","--mine","--page-size","1","--format","json"],
                             env=env,capture_output=True,text=True,timeout=90,check=False)
            if p.returncode==0:
                report["account_verified"]=True
                report["result"]="authorized_private_kernel_listing_ok"
            else:
                report["result"]=classify((p.stderr or "")+"\n"+(p.stdout or ""))
        except subprocess.TimeoutExpired:
            report["result"]="cli_timeout_no_retry"
        except (OSError,ValueError):
            report["result"]="cli_execution_unavailable"
    # Save only safe enums; no account identifiers, usernames, outputs, or contents.
    target=os.getenv("TFLEX_KAGGLE_DIAGNOSTIC_REPORT","")
    if target:
        path=Path(target);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("TFLEX_KAGGLE_ACCOUNT_DIAGNOSTIC",json.dumps(report,sort_keys=True))
    return 0 if report["account_verified"] else 2

if __name__=="__main__":
    sys.exit(run())
