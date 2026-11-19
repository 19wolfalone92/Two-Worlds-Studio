"""Safely check TWO separate legitimate Kaggle token identities. Never reveal names."""
import json, os, sys
from pathlib import Path
import kagglehub

def resolve(secret):
    if not secret:
        raise RuntimeError("missing_gitHub_secret")
    previous=os.environ.get("KAGGLE_API_TOKEN")
    try:
        os.environ["KAGGLE_API_TOKEN"]=secret
        identity=kagglehub.auth.whoami(verbose=False)
        name=identity.get("username") if isinstance(identity,dict) else None
        if not isinstance(name,str) or len(name)<2:
            raise RuntimeError("identity_not_verified")
        return name.casefold()
    finally:
        if previous is None:os.environ.pop("KAGGLE_API_TOKEN",None)
        else:os.environ["KAGGLE_API_TOKEN"]=previous

def main():
    # Secrets are passed in separately by Github. No values ever printed.
    first=os.getenv("TFLEX_ACCOUNT_ONE_KGAT","")
    second=os.getenv("TFLEX_ACCOUNT_TWO_KGAT","")
    if not(first and second):
        result={"distinct_account_proof":False,"status":"missing_secret","gpu_launched":False}
        status=2
    else:
        try:
            u1=resolve(first);u2=resolve(second)
            result={"distinct_account_proof":u1!=u2,
                    "status":"two_authorized_distinct_accounts" if u1!=u2 else "same_account",
                    "gpu_launched":False,"original_images_shared":False}
            status=0 if u1!=u2 else 2
        except Exception as exc:
            result={"distinct_account_proof":False,"status":"identity_query_failed",
                    "safe_error_class":type(exc).__name__,"gpu_launched":False}
            status=2
    print("TFLEX_TWO_KAGGLE_IDENTITIES",json.dumps(result,sort_keys=True))
    path=os.getenv("TFLEX_DISTINCT_REPORT")
    if path:
        p=Path(path);p.parent.mkdir(exist_ok=True,parents=True)
        p.write_text(json.dumps(result,indent=2))
    return status

if __name__=="__main__":
    sys.exit(main())
