"""T-Flex: confidential Kaggle Python API smoke check (never launches a kernel).

The official KaggleHub identity is authenticated independently. This script
checks whether Kaggle's Python API can LIST ONE owned kernel, capturing
all third-party stdout/stderr so account names and token never enter CI logs.
Only sanitized enums are written to one safe GitHub Actions artifact.
"""
import contextlib,io,json,os,sys
from pathlib import Path

def classify_exception(error):
    typ=type(error).__name__
    resp=getattr(error,"response",None)
    status=getattr(resp,"status_code",None)
    if isinstance(status,int) and 100<=status<=599:
        return "http_"+str(status)
    if typ in ("TimeoutExpired","Timeout","ConnectTimeout","ReadTimeout"):return "timeout"
    if typ in ("SSLError","ConnectionError"):return "network"
    if typ in ("ValueError","TypeError","KeyError","AttributeError","RuntimeError","ImportError","SystemExit"):
        return "python_"+typ.lower()
    return "other_safe_exception"

def run():
    res={"kagglehub_identity":False,"kaggle_python_api_authenticated":False,
         "kaggle_python_api_kernel_listing":False,
         "kaggle_gpu_started":False,"error":"not_checked"}
    secret=os.environ.get("KAGGLE_API_TOKEN","")
    if not secret:
        res["error"]="missing_secret"
    else:
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                import kagglehub
                identity=kagglehub.auth.whoami(verbose=False)
                username=identity.get("username")
            res["kagglehub_identity"]=bool(username)
        except BaseException as error:
            res["error"]="kagglehub_"+classify_exception(error)
        if res["kagglehub_identity"]:
            try:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    client=KaggleApi()
                    client.authenticate()
                res["kaggle_python_api_authenticated"]=True
                res["error"]="none"
                try:
                    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                        data=client.kernels_list(mine=True,page_size=1)
                    res["kaggle_python_api_kernel_listing"]=isinstance(data,(list,type(None)))
                    res["error"]="none" if res["kaggle_python_api_kernel_listing"] else "unexpected_response_structure"
                except BaseException as error:
                    res["error"]=classify_exception(error)
            except BaseException as error:
                res["error"]=classify_exception(error)
    print("TFLEX_KAGGLE_PYTHON_API_SAFE",json.dumps(res,sort_keys=True))
    out=os.environ.get("TFLEX_PYTHON_API_SAFE_REPORT")
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(res,indent=2))
    return 0 if res["kaggle_python_api_kernel_listing"] else 2

if __name__=="__main__":sys.exit(run())
