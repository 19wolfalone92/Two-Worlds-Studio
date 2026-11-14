"""Private Kaggle GPU kernel forensic. Diagnostic-only; never resubmit or publish.
No user photos, voices, prompts, or raw logs leave the private Kaggle runner.
"""
from __future__ import annotations
import json,os,re,subprocess,tempfile,hashlib
from pathlib import Path

KERNEL="wolf19loner92/eb01-ltx2b-pilot33-fd6d11a64bd6"
SAFE_STAGE=re.compile(r"EB01_STAGE_([A-Z0-9_]{3,64})(?:\s|$)")
SAFE_EXCEPTION=re.compile(r"(?:^|[\s:])(RuntimeError|ValueError|AttributeError|TypeError|KeyError|AssertionError|FileNotFoundError|ModuleNotFoundError|OutOfMemoryError|ImportError|SyntaxError|TimeoutExpired|MemoryError|OSError|ConnectionError)(?:[:\s]|$)")
SAFE_PHRASES=("exceeded 1400s","CUDA out of memory","float16","cpu","GPU unavailable","No module named","No unique actual generated video","Official LTX","source changed")
def run(args,timeout=100):
    try:
        r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
        return (r.returncode,r.stdout,r.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:
        return (99,"",type(e).__name__)
def redact(text):
    text=re.sub(r"(?i)(hf_[a-zA-Z0-9]{10,}|sk-[a-zA-Z0-9_-]{12,})","[TOKEN]",text)
    text=re.sub(r"(/(?:kaggle|home|tmp|mnt|root|input|working)/[^\s'\"]+)","[PRIVATE_PATH]",text)
    text=re.sub(r"(?i)(api[_ -]?key|authorization|password|bearer|token)\s*[=:]\s*[^\s,]+",r"\1=[REDACTED]",text)
    text=re.sub(r"https?://[^\s]+","[URL]",text)
    return text[:200]
def main():
    report={"schema":1,"original_kernel":KERNEL,"read_only":True,"gpu_restarted":False,"published":False,"whole_film_ready":False}
    code,status,stderr=run(["kaggle","kernels","status",KERNEL])
    # Kaggle CLI may emit status text on stderr instead of stdout.
    match=re.search(r"KernelWorkerStatus\.(QUEUED|RUNNING|COMPLETE|ERROR|CANCEL)",status+"\n"+stderr)
    report["state"]=match.group(1) if match else "UNKNOWN"
    report["status_cli_exit_code"]=code
    if report["state"]=="UNKNOWN":
        hint=(status+"\n"+stderr).lower()
        report["status_api_error_category"]=(
            "RATE_LIMITED" if "429" in hint else
            "AUTH_FAILED" if any(x in hint for x in ("401","403","unauthorized","forbidden")) else
            "KAGGLE_STATUS_CLI_UNAVAILABLE")
    
    if report["state"] not in ("ERROR","COMPLETE","CANCEL"):
        print("EB01_PRIVATE_FORENSIC",json.dumps(report,sort_keys=True))
        return
    with tempfile.TemporaryDirectory(prefix="eb01-private-") as td:
        output=Path(td)
        ret,_,_=run(["kaggle","kernels","output",KERNEL,"-p",str(output),"--force"],timeout=210)
        report["private_output_readback_ok"]=ret==0
        receipt=output/"EB01_S02_LTX_PILOT33_JOB.json"
        if receipt.exists():
            try:
                d=json.loads(receipt.read_text(encoding="utf-8"))
                report["receipt"]={k:d.get(k) for k in ("schema","success","status","shot","error_type","frame_count","clip_sha256")}
                msg=str(d.get("error_summary",""))
                report["error_summary_sanitized"]=redact(msg)
            except (ValueError,UnicodeError):
                report["receipt_parse_failed"]=True
        else:
            report["receipt_missing"]=True
        code,logs,_=run(["kaggle","kernels","logs",KERNEL],timeout=90)
        gpu=(output/"EB01_LTX_PRIVATE_GPU_LOG.txt")
        combined=(gpu.read_text(errors="replace")[-110000:] if gpu.exists() else "")+"\n"+logs[-150000:]
        report["error_log_sha256"]=hashlib.sha256(combined.encode()).hexdigest()
        stages=SAFE_STAGE.findall(combined)
        report["real_stages_observed"]=sorted(set(stages))
        report["last_native_stage"]=stages[-1] if stages else "NONE"
        types=SAFE_EXCEPTION.findall(combined)
        report["actual_exception_classes"]=list(dict.fromkeys(types))[-15:]
        report["known_failure_patterns_present"]={t: t.lower() in combined.lower() for t in SAFE_PHRASES}
        report["gpu_private_log_bytes"]=gpu.stat().st_size if gpu.exists() else 0
        report["video_exists"]=any(p.suffix.lower()==".mp4" for p in output.rglob("*.mp4"))
        # Last diagnostic lines strictly redacted, no private prompt/data.
        safe=[]
        for line in combined.splitlines():
            if re.search(r"(?i)(EB01_STAGE_|EB01_ALT_MODEL_INFERENCE_BLOCKED|EB01_LTX_PILOT_TIMEOUT|RuntimeError:|AttributeError:|TypeError:|AssertionError:|ModuleNotFoundError:|OutOfMemoryError:|FileNotFoundError:|TimeoutExpired:)",line):
                # Full serialized lines can contain scene input. Never output those.
                if "LTX_PROMPT" in line or "input/" in line or '"prompt"' in line:continue
                safe.append(redact(line))
        report["sanitized_trace_tail"]=safe[-14:]
    print("EB01_PRIVATE_FORENSIC",json.dumps(report,sort_keys=True,ensure_ascii=False))
    path=Path(os.environ["RUNNER_TEMP"])/"EB01_NEW_GPU_REAL_ERROR_SUMMARY.json"
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
if __name__=="__main__": main()
