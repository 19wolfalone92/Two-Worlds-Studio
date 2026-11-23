"""Read-only Kaggle Studio-wave forensic; no GPU, no requeue, no user media returned."""
from __future__ import annotations
import json,re,subprocess,tempfile,hashlib
from pathlib import Path

KERNEL="wolf19loner92/studio-native-man-wave-81a7215251acc6b"
TYPES=re.compile(r"(?<![A-Za-z])(SyntaxError|NameError|TypeError|ModuleNotFoundError|ImportError|RuntimeError|ValueError|AssertionError|FileNotFoundError|OSError|AttributeError|CUDAOutOfMemoryError|OutOfMemoryError|MemoryError|TimeoutExpired|PermissionError)(?![A-Za-z])")
STAGES=re.compile(r"(?:STUDIO_WAVE_|WAVE_STAGE_)[A-Z0-9_]{3,80}")
def call(args,timeout=140):
    try:
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         text=True,timeout=timeout)
        return p.returncode,p.stdout,p.stderr
    except (OSError,subprocess.TimeoutExpired) as err:
        return 101,"",type(err).__name__
def main():
    result={"schema":1,"kernel":KERNEL,"gpu_submissions":0,"read_only":True,
            "new_gpu_inference":False,"public_media_uploaded":False,
            "video_artistic_approved":False}
    rc,status,err=call(["kaggle","kernels","status",KERNEL])
    m=re.search(r"KernelWorkerStatus\.(QUEUED|RUNNING|ERROR|COMPLETE|CANCEL)",status+"\n"+err)
    result["state"]=m.group(1) if m else "UNKNOWN"
    result["status_code"]=rc
    with tempfile.TemporaryDirectory(prefix="studio-wave-forensic-private-") as d:
        target=Path(d)
        output_rc,_,_=call(["kaggle","kernels","output",KERNEL,"-p",str(target),"--force"],timeout=240)
        result["output_read_ok"]=output_rc==0
        result["output_file_count"]=len(list(target.iterdir()))
        logs_rc,logs,logs_err=call(["kaggle","kernels","logs",KERNEL],timeout=150)
        result["logs_read_ok"]=logs_rc==0
        internal='\n'.join(p.read_text(errors="replace")[-90000:]
              for p in target.glob("STUDIO_WAVE_PRIVATE_GPU*.log"))
        combined=(internal+"\n"+logs+"\n"+logs_err)[-150000:]
        result["logs_sha256"]=hashlib.sha256(combined.encode()).hexdigest()
        all_stages=STAGES.findall(combined)
        result["stages"]=list(dict.fromkeys(all_stages))[-30:]
        result["last_stage"]=all_stages[-1] if all_stages else "NONE"
        result["exception_classes"]=list(dict.fromkeys(TYPES.findall(combined)))[-20:]
        result["gpu_report_found"]=any(x.suffix==".mp4" for x in target.iterdir())
        result["error_markers"]={
            k:k.lower() in combined.lower() for k in (
                "No module named","SyntaxError","ModuleNotFoundError","NameError",
                "CUDA out of memory","OutOfMemoryError","checkpoint",
                "download","ConnectionError","official LTX revision changed",
                "STUDIO_WAVE_PRIVATE_REFERENCE","source revision mismatch",
                "No such file or directory","Hugging Face","GPU unavailable",
                "tokenizer","vram")}
        receipt=target/"STUDIO_WAVE_TRUE_NATIVE_FAILURE_SAFE.json"
        if receipt.is_file():
            try:
                obj=json.loads(receipt.read_text())
                result["model_failure"]={k:obj.get(k) for k in ("stage","error_type","successful_native_segments")}
            except (UnicodeError,ValueError):
                result["model_failure_invalid"]=True
        # Do NOT expose traceback lines / prompts / encrypted reference bytes in public GH logs.
        result["traceback_frame_lines_count"]=sum(
          1 for line in combined.splitlines() if line.strip().startswith("File "))
        safe_errors=[]
        for line in combined.splitlines():
            if "Traceback" in line or "Error:" in line or "Exception:" in line:
                line=re.sub(r"(/(?:kaggle|root|tmp|home|input|working)/[^\s,\"']+)","[PRIVATE_PATH]",line)
                line=re.sub(r"(https?://\S+)","[REDACTED_URL]",line)
                line=re.sub(r"[A-Za-z0-9+/]{90,}={0,2}","[BASE64_REDACTED]",line)
                if any(x in line for x in ("STUDIO_WAVE_IMAGE","LTX_PROMPT","token","Authorization")):
                    continue
                safe_errors.append(line[-220:])
        result["safe_error_lines"]=safe_errors[-8:]
    print("STUDIO_NATIVE_WAVE_PRIVATE_FORENSIC",json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
