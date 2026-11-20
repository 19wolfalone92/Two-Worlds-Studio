"""Read known private Kaggle pilot logs and emit ONLY a sanitized failure class.
No screenshots, tokens, notebook body, account names or raw exception messages.
No GPU is launched.
"""
from __future__ import annotations
import contextlib,io,json,os,re,sys
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

def classify(text):
    t=text.lower()
    checks=[("module_import_failure","no module named"),
            ("git_clone_failure","fatal: unable to access"),
            ("git_fetch_failure","couldn't find remote ref"),
            ("pip_dependency_conflict","resolutionimpossible"),
            ("pip_install_error","subprocess.calledprocesserror"),
            ("pip_ssl_error","certificate verify failed"),
            ("pytorch_out_of_memory","cuda out of memory"),
            ("cuda_not_found","cuda gpu unavailable"),
            ("hf_model_auth","gatedrepoerror"),
            ("huggingface_403","403 client error"),
            ("huggingface_404","404 client error"),
            ("hf_network_timeout","timed out"),
            ("flashvdm_missing","flashvdm"),
            ("transformer_import_error","transformers"),
            ("pymeshlab_error","pymeshlab"),
            ("subprocess_timeout","timeoutexpired"),
            ("python_import_error","importerror"),
            ("python_syntax_error","syntaxerror")]
    for name,pat in checks:
        if pat in t:return name
    return "other_private_kaggle_error"

def main():
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            who=kagglehub.auth.whoami(verbose=False)["username"]
            api=KaggleApi();api.authenticate()
            raw=api.kernels_logs(who+"/tflex-v94-hunyuan-official-public-sample")
        result={"log_retrieved":True,"log_chars":len(raw),
                "failure_category":classify(raw),"user_data_transferred":False}
    except BaseException as exc:
        result={"log_retrieved":False,"safe_error_type":type(exc).__name__,
                "user_data_transferred":False}
    print("TFLEX_V94_KAGGLE_PRIVATE_FAILURE_CATEGORY",json.dumps(result,sort_keys=True))
    if not result["log_retrieved"]:sys.exit(2)

if __name__=="__main__":main()
