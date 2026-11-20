"""Read-only, private Kaggle GPU proof notebook discovery.

After a 409 conflict, never blindly submit another notebook version.
We verify whether a job exists and whether its progress is readable.
Only boolean/status/error category is printed to public CI; account handle hidden.
"""
import contextlib,io,json,os,sys
from pathlib import Path

NAME="tflex-v9-200mm-gpu-hardware-check"

def safe_exc(e):
    resp=getattr(e,'response',None)
    code=getattr(resp,'status_code',None)
    return "http_"+str(code) if isinstance(code,int) and 100<=code<=599 else type(e).__name__

def main():
    report={"private_kernel_listed":False,"status_readable":False,
      "status":"unverified","error":"not_checked","gpu_proved":False,
      "requeued_any_gpu_job":False,"user_originals_copied":False}
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            import kagglehub
            from kaggle.api.kaggle_api_extended import KaggleApi
            who=kagglehub.auth.whoami(verbose=False).get("username")
            if not who:raise ValueError("anonymous")
            api=KaggleApi();api.authenticate()
            listing=api.kernels_list(mine=True,page_size=100) or []
        kernel=who+"/"+NAME
        refs=[str(getattr(k,'ref','') or '') for k in listing]
        report["private_kernel_listed"]=any(r.lower()==kernel.lower() for r in refs)
        if report["private_kernel_listed"]:
            try:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    info=api.kernels_status(kernel)
                # Public status category only; never print raw API objects.
                candidate=str(getattr(info,"status","")).upper()
                if candidate:
                    report["status_readable"]=True
                    report["status"]=next((s for s in ("COMPLETE","ERROR","RUNNING","QUEUED","NEW_SCRIPT")
                                          if s in candidate),"unknown")
                    report["error"]="none"
            except BaseException as e:
                report["error"]=safe_exc(e)
        else:report["error"]="private_kernel_not_found_in_page_one"
    except BaseException as e:
        report["error"]=safe_exc(e)
    print("TFLEX_EXISTING_PRIVATE_GPU_KERNEL_SAFE",json.dumps(report,sort_keys=True))
    p=os.environ.get("TFLEX_KAGGLE_EXISTING_STATUS_REPORT")
    if p:
        output=Path(p);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2))
    return 0 if report["private_kernel_listed"] else 2

if __name__=="__main__":sys.exit(main())
