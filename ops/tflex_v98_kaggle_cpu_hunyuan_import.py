"""Kaggle CPU test: load EXACT official Hunyuan3D-2mv source, no weights/GPU.
Leaves safe status in private Kaggle result even after a Python import error.
"""
import json,subprocess,sys,os,time
from pathlib import Path

def work():
    out=Path("/kaggle/working/tflex_v98_cpu_official_hunyuan_import.json")
    data={"git_pinned_repo_checkout":False,"required_minimal_wheels_installed":False,
          "Hunyuan3D_shape_pipeline_imported":False,"gpu_used":False,
          "user_screenshots_uploaded":False,"failure_phase":None,"failure_type":None}
    phase="start"
    try:
        path=Path("/kaggle/working/Hunyuan3D-2")
        if not path.is_dir():
            phase="git_clone"
            subprocess.run(["git","clone","--quiet","--depth","1",
                "https://github.com/Tencent-Hunyuan/Hunyuan3D-2",str(path)],
                check=True,capture_output=True,timeout=180)
        phase="git_fetch_pinned_revision"
        subprocess.run(["git","-C",str(path),"fetch","--quiet","--depth","1",
            "origin","f8db63096c8282cb27354314d896feba5ba6ff8a"],
            check=True,capture_output=True,timeout=180)
        subprocess.run(["git","-C",str(path),"checkout","--quiet","--detach",
            "f8db63096c8282cb27354314d896feba5ba6ff8a"],
            check=True,capture_output=True,timeout=60)
        data["git_pinned_repo_checkout"]=True
        phase="pip_minimal"
        subprocess.run([sys.executable,"-m","pip","install","-q",
             "trimesh>=4.5","omegaconf","einops","accelerate>=1.0"],
            check=True,capture_output=True,timeout=320)
        data["required_minimal_wheels_installed"]=True
        phase="import_hy3d"
        sys.path.insert(0,str(path))
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        data["Hunyuan3D_shape_pipeline_imported"]=bool(Hunyuan3DDiTFlowMatchingPipeline)
    except BaseException as exc:
        data["failure_phase"]=phase
        data["failure_type"]=type(exc).__name__
    finally:
        out.write_text(json.dumps(data,indent=2))
        print("TFLEX_V98_CPU_HUNYUAN_IMPORT_STATUS",json.dumps(data,sort_keys=True))
if __name__=="__main__":work()
