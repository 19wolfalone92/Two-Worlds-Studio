"""Read-only studio connection audit. Never log token values or response bodies."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
from urllib import error, request

# Secrets are passed into env explicitly by GitHub Actions and never printed.
SERVICES = {
    "KAGGLE": ("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_2"),
    "HUGGING_FACE": ("HF_TOKEN", "HUGGINGFACE_TOKEN"),
    "CLOUDFLARE_AI": ("CLOUDFLARE_AI_TOKEN", "CLOUDFLARE_API_TOKEN", "CF_API_TOKEN"),
    "GROQ": ("GROQ_API_KEY",),
    "GEMINI": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "OPENROUTER": ("OPENROUTER_API_KEY",),
    "AI_HORDE": ("HORDE_API_KEY", "AI_HORDE_API_KEY"),
    "AZURE_SPEECH": ("AZURE_SPEECH_KEY",),
    "R2_ACCESS": ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"),
    "MODAL": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
}
VARIABLES = {
    "CLOUDFLARE_ACCOUNT_ID": ("CLOUDFLARE_ACCOUNT_ID",),
    "D1_DATABASE_ID": ("CLOUDFLARE_D1_DATABASE_ID",),
    "CLOUDFLARE_QUEUE": ("CLOUDFLARE_QUEUE_NAME",),
    "AZURE_SPEECH_REGION": ("AZURE_SPEECH_REGION",),
    "R2_BUCKET": ("R2_BUCKET_NAME",),
}
def http_probe(url, headers, timeout=12):
    try:
        req = request.Request(url, headers=headers, method="GET")
        with request.urlopen(req, timeout=timeout) as response:
            # Don't read or print response body; success indicates API accepted credentials.
            return "AUTH_OK", "HTTP_"+str(response.status)
    except error.HTTPError as exc:
        if exc.code == 429:
            return "RATE_LIMITED", "HTTP_429_AUTH_UNDETERMINED"
        if exc.code in (401, 403):
            return "AUTH_REJECTED_OR_FORBIDDEN", "HTTP_"+str(exc.code)
        return "API_ERROR", "HTTP_"+str(exc.code)
    except Exception:
        return "NETWORK_OR_TIMEOUT", "No safe authentication result"

def locate(names):
    return [(name, os.environ.get(name, "").strip()) for name in names if os.environ.get(name, "").strip()]

def probe(service, detected):
    if not detected:
        return service, "NOT_CONFIGURED_UNDER_EXPECTED_NAME", "No matching configured secret"
    token = detected[0][1]
    try:
        if service == "KAGGLE":
            p = subprocess.run(["kaggle","quota","--format","json"], capture_output=True,
                               text=True, timeout=40, check=False)
            if p.returncode == 0:
                return service, "AUTH_OK", "read_only_quota"
            if "429" in p.stderr or "Too Many Requests" in p.stderr:
                return service, "RATE_LIMITED", "HTTP_429"
            return service, "CHECK_FAILED", "CLI auth or quota check failed"
        if service == "HUGGING_FACE":
            st, note = http_probe("https://huggingface.co/api/whoami-v2",
                {"Authorization": "Bearer " + token})
        elif service == "CLOUDFLARE_AI":
            st, note = http_probe("https://api.cloudflare.com/client/v4/user/tokens/verify",
                {"Authorization": "Bearer " + token})
        elif service == "GROQ":
            st, note = http_probe("https://api.groq.com/openai/v1/models",
                {"Authorization": "Bearer " + token})
        elif service == "GEMINI":
            st, note = http_probe("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
                {"x-goog-api-key": token})
        elif service == "OPENROUTER":
            st, note = http_probe("https://openrouter.ai/api/v1/auth/key",
                {"Authorization": "Bearer " + token})
        elif service == "AI_HORDE":
            st, note = http_probe("https://stablehorde.net/api/v2/find_user",
                {"apikey": token})
        elif service == "AZURE_SPEECH":
            region=os.environ.get("AZURE_SPEECH_REGION","").strip()
            if not region:
                return service, "MISSING_REGION", "AZURE_SPEECH_REGION variable required"
            if not all(c.isalnum() or c=="-" for c in region):
                return service, "INVALID_REGION_FORMAT", "Region format not valid"
            st,note = http_probe("https://"+region+".tts.speech.microsoft.com/cognitiveservices/voices/list",
                {"Ocp-Apim-Subscription-Key": token})
        else:
            return service, "PRESENT_NOT_AUTHENTICATED", "No read-only endpoint tested"
        return service, st, note
    except Exception:
        return service, "CHECK_FAILED", "Exception suppressed to protect credentials"

def main():
    matches = {name: locate(aliases) for name, aliases in SERVICES.items()}
    # Print ONLY exact secret names, not secret contents, lengths, prefixes, or hashes.
    print("STUDIO_CONNECTION_AUDIT=v1")
    for name, items in matches.items():
        print("SECRET", name, "FOUND_NAMES="+(",".join(i[0] for i in items) if items else "NONE"))
    for label,aliases in VARIABLES.items():
        present = [x for x in aliases if os.environ.get(x,"").strip()]
        print("VARIABLE",label,"CONFIGURED" if present else "MISSING")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        future_map={pool.submit(probe,name,items):name for name,items in matches.items()}
        results={}
        for future in concurrent.futures.as_completed(future_map):
            service=future_map[future]
            try:
                results[service]=future.result()
            except Exception:
                results[service]=(service,"CHECK_FAILED","Safe diagnostic failed")
    for name in SERVICES:
        service,status,note=results[name]
        print("RESULT",service,status,note)
    print("AUDIT_COMPLETE; read-only probes; no AI inference, no GPU, no paid generation")

if __name__=="__main__":
    main()
