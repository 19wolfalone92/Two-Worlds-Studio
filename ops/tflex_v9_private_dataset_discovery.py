"""Private read-only check: do user's authenticated Kaggle accounts already
contain a V9.1 figurine-reference dataset? Never print private dataset IDs.
Only report counts of matching dataset titles/refs, no images copied.
"""
from __future__ import annotations
import contextlib,io,json,os,sys
from pathlib import Path

def run():
    r={"authenticated":False,"searched_own_private_datasets":False,
       "matching_figurine_reference_datasets":0,"authorization_to_open_source":False,
       "gpu_launched":False,"original_images_copied":False,
       "error":"not_attempted"}
    if os.environ.get("KAGGLE_API_TOKEN"):
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                from kaggle.api.kaggle_api_extended import KaggleApi
                api=KaggleApi();api.authenticate()
                r["authenticated"]=True
                data=api.dataset_list(mine=True,page=1) or []
            r["searched_own_private_datasets"]=True
            words=("winged","tflex","character-v9","figurine","sculpt","200mm")
            refs=[str(getattr(d,"ref","") or "")+" "+str(getattr(d,"title","") or "")
                  for d in data]
            r["matching_figurine_reference_datasets"]=sum(
                 any(k in ref.casefold() for k in words) for ref in refs)
            r["error"]="none"
        except Exception as exc:
            e=type(exc).__name__
            r["error"]=e if e in ("HTTPError","ValueError","RuntimeError","Timeout","ConnectionError") else "sanitized_api_failure"
    print("TFLEX_PRIVATE_KAGGLE_DATASET_SCAN",json.dumps(r,sort_keys=True))
    p=os.environ.get("TFLEX_KAGGLE_DATASET_SCAN_OUT")
    if p:
        out=Path(p);out.parent.mkdir(exist_ok=True,parents=True)
        out.write_text(json.dumps(r,sort_keys=True))
    return 0 if r["searched_own_private_datasets"] else 2

if __name__=="__main__":sys.exit(run())
