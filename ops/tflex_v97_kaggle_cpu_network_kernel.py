"""CPU-only Kaggle execution test for official-public 3D model access.
No user-original screenshots, no model training, no GPU quota, safe status only.
"""
import importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path

def probe_command(args,limit):
    try:
        p=subprocess.run(args,check=False,capture_output=True,text=True,timeout=limit)
        return {"executed":True,"returncode_zero":p.returncode==0,
                "timeout":False}
    except subprocess.TimeoutExpired:
        return {"executed":True,"returncode_zero":False,"timeout":True}
    except BaseException:
        return {"executed":False,"returncode_zero":False,"timeout":False}

def main():
    status={"project":"TFLEX V97 publicly sourced Hunyuan preflight",
         "private_user_images_received":False,
         "source_user_dataset_mounted":False,
         "real_3d_mesh_generated":False,"GPU_seconds":0,
         "environment_package_presence":{},
         "git_lsremote":{},"huggingface_head":{},
         "error_type":"none"}
    try:
        status["free_disk_gib"]=round(shutil.disk_usage("/kaggle/working").free/2**30,2)
        for name in ["torch","numpy","trimesh","diffusers",
                     "transformers","accelerate","safetensors",
                     "scipy","PIL","pymeshlab","xatlas"]:
            status["environment_package_presence"][name]=importlib.util.find_spec(name)!=None
        status["git_lsremote"]=probe_command(["git","ls-remote","-q",
             "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git","HEAD"],35)
        import requests
        try:
            resp=requests.head("https://huggingface.co/tencent/Hunyuan3D-2mv",
                               timeout=25,allow_redirects=True)
            status["huggingface_head"]={"executed":True,
                "http_status":resp.status_code if resp.status_code in
                   (200,301,302,401,403,404,429,500,502,503)
                 else "other"}
        except BaseException as exc:
            status["huggingface_head"]={"executed":False,"safe_error_type":type(exc).__name__}
    except BaseException as exc:
        status["error_type"]=type(exc).__name__
    finally:
        path=Path("/kaggle/working/tflex_v97_cpu_public_model_network_preflight.json")
        path.write_text(json.dumps(status,sort_keys=True,indent=2))
        print("TFLEX_V97_CPU_PREFLIGHT",json.dumps(status,sort_keys=True))
if __name__=="__main__":main()
