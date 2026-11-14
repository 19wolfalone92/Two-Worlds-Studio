"""Strictly read-only two-Kaggle-account access verifier.

Works from env-protected GitHub Actions secrets; prints NO tokens or handles.
It never sends studio original private pictures/voices to the second account.
"""
from __future__ import annotations
import json,os,re,subprocess
from pathlib import Path

HANDLE=re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,29}$")
def config_name(output: str) -> str | None:
    m=re.search(r"(?im)^\s*-?\s*username\s*:\s*([A-Za-z0-9_-]{2,30})\s*$",output)
    return m.group(1).lower() if m and HANDLE.fullmatch(m.group(1)) else None

def query(token: str, runner=subprocess.run) -> tuple[str,str|None]:
    if not token:
        return "MISSING_GITHUB_SECRET",None
    env=os.environ.copy()
    env["KAGGLE_API_TOKEN"]=token
    env.pop("KAGGLE_USERNAME",None)
    try:
        cli=runner(["kaggle","config","view"],env=env,text=True,
            capture_output=True,timeout=45)
        if cli.returncode:
            return "TOKEN_FORMAT_OR_CLI_REJECTED",None
        name=config_name(cli.stdout)
        if not name:
            return "ACCOUNT_NAME_NOT_RESOLVED",None
        # This ONE read-only network query proves token is usable; metadata
        # about private datasets/assets is never printed or transferred.
        ping=runner(["kaggle","kernels","list","--mine","--page-size","1"],
            env=env,text=True,capture_output=True,timeout=75)
        if ping.returncode:
            return "KAGGLE_AUTH_OR_API_ACCESS_FAILED",name
        return "AUTHENTICATED_READ_ONLY",name
    except (OSError,subprocess.TimeoutExpired):
        return "KAGGLE_API_UNAVAILABLE",None

SOURCE_DATASET = "wolf19loner92/eb01-v7-private-production-sources"

def verify_brother_source_access(token: str, *, runner=subprocess.run) -> str:
    """Only metadata access. Never download original portraits/WAVs."""
    if not token:
        return "MISSING_SECRET"
    env=os.environ.copy()
    env["KAGGLE_API_TOKEN"]=token
    env.pop("KAGGLE_USERNAME",None)
    try:
        response=runner(
            ["kaggle","datasets","files",SOURCE_DATASET],
            env=env,text=True,capture_output=True,timeout=75)
    except (OSError,subprocess.TimeoutExpired):
        return "SOURCE_ACL_PROBE_INFRA_UNAVAILABLE"
    if response.returncode == 0:
        return "ORIGINAL_OWNER_PRIVATE_DATASET_METADATA_VISIBLE"
    body=((response.stderr or "")+"\\n"+(response.stdout or "")).lower()
    if "429" in body or "too many requests" in body:
        return "KAGGLE_RATE_LIMITED_DO_NOT_RETRY"
    if "403" in body or "forbidden" in body or "permission" in body:
        return "PRIVATE_SOURCE_ACCESS_DENIED"
    if "404" in body or "not found" in body:
        return "PRIVATE_SOURCE_DATASET_NOT_VISIBLE"
    return "PRIVATE_SOURCE_ACCESS_NOT_CONFIRMED"

def check(primary: str, brother: str,runner=subprocess.run) -> dict:
    first,n1=query(primary,runner=runner)
    second,n2=query(brother,runner=runner)
    distinct=bool(first=="AUTHENTICATED_READ_ONLY" and
                  second=="AUTHENTICATED_READ_ONLY" and n1!=n2)
    source_status=(verify_brother_source_access(brother,runner=runner)
        if distinct else "NOT_CHECKED_INDEPENDENT_AUTH_REQUIRED")
    source_ok=(source_status=="ORIGINAL_OWNER_PRIVATE_DATASET_METADATA_VISIBLE")
    return {
        "schema":1,"original_account":first,"brother_account":second,
        "two_distinct_authenticated_accounts":distinct,
        "brother_gpu_connected":False,
        "brother_private_studio_source_access_verified":source_ok,
        "private_source_access_status":source_status,
        "originals_copied":False,"gpu_inference_launched":False,
        "used_paid_services":False,"published":False,
        "next_action":("VERIFY_BROTHER_GPU_AND_LTX_PATCHED_PILOT_ONLY"
            if source_ok else ("VERIFY_OWNER_APPROVED_PRIVATE_DATASET_SHARING_NO_RAW_COPY"
            if distinct else "ADD_KAGGLE_API_TOKEN2_TO_THIS_REPOSITORY_IF_MISSING_OR_FIX_AUTH")),
    }

def main():
    result=check(os.getenv("PRIMARY_KAGGLE_TOKEN",""),
                 os.getenv("BROTHER_KAGGLE_TOKEN",""))
    path=Path(os.environ["GITHUB_OUTPUT_REPORT"])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print("EB01_TWO_ACCOUNTS_AUTH_AUDIT",json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
