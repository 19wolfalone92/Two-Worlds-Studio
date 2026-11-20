"""Read private owner Hunyuan Kaggle failure_message field, print ONLY safe tags.
No new GPU, no usernames/tokens/logs, no original screenshots.
"""
import contextlib,io,json,os,sys
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

def classify(message):
    m=str(message or "").lower()
    keys={
       "git_clone":"git", "pip":"pip", "dependency":"dependency",
       "memory":"memory", "disk":"disk", "cuda":"cuda", "torch":"torch",
       "internet":"internet", "network":"network", "ssl":"ssl",
       "timeout":"timeout","source":"source",
       "gpu":"gpu","quota":"quota","kernel":"kernel","image":"image",
       "permission":"permission","install":"install","import":"import",
       "error":"error","fail":"fail","exit":"exit", "python":"python",
       "resource":"resource","gradio":"gradio", "hf":"huggingface",
       "license":"license", "runtime":"runtime","preemption":"preempt",
       "package":"package", "requirements":"requirements",
       "exception":"exception", "notfound":"not found",
       "version":"version", "notebook":"notebook"}
    return [name for name,needle in keys.items() if needle in m]

def main():
    result={"owner_private_hunyuan_state":"unknown","failure_message_present":False,
            "failure_message_length":0,"safe_tags":[],"gpu_launched":False}
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            who=kagglehub.auth.whoami(verbose=False).get("username")
            api=KaggleApi();api.authenticate()
            x=api.kernels_status(who+"/tflex-v94-hunyuan-official-public-sample")
        status=str(getattr(x,"status",""))
        result["owner_private_hunyuan_state"]=(
          "ERROR" if "ERROR" in status.upper() else
          "COMPLETE" if "COMPLETE" in status.upper() else
          "RUNNING" if "RUNNING" in status.upper() else "OTHER")
        message=str(getattr(x,"failure_message","") or "")
        result["failure_message_present"]=bool(message)
        result["failure_message_length"]=len(message)
        result["safe_tags"]=classify(message)
    except BaseException as exc:
        result["safe_error_class"]=type(exc).__name__
    print("TFLEX_V94_PRIVATE_SESSION_FAILURE_SAFE",json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
