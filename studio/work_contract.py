"""Strict boundary: ChatGPT Work authors/scripts; TWO WORLDS Studio never rewrites them.

A script fingerprint prevents accidentally changing an approved scene while
compositing/animating. The owner is responsible for recording real approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from studio.continuity import check_episode

SCRIPT_FIELDS=("id","duration","palette","effect","heading","caption","shot",
               "voiceover","continuity","subjects","secondary_speakers")
def script_fingerprint(ep:dict)->str:
    if not isinstance(ep.get("scenes"),list) or not ep["scenes"]:
        raise ValueError("No scenes in approved Work script")
    payload={"id":ep.get("id"),"title":ep.get("title"),
             "scenes":[{key:s[key] for key in SCRIPT_FIELDS if key in s}
                       for s in ep["scenes"]]}
    canonical=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_work_lock(ep:dict,*,strict:bool=True)->dict:
    check_episode(ep,strict=True)
    spec=ep.get("work_lock")
    if not isinstance(spec,dict):
        if strict:
            raise ValueError("No Work-authored episode lock; preview only")
        return {"work_lock_valid":False,"script_sha256":script_fingerprint(ep)}
    if spec.get("origin")!="chatgpt_work":
        raise ValueError("Origin must be chatgpt_work for production lock")
    expected=script_fingerprint(ep)
    if spec.get("script_sha256")!=expected:
        raise ValueError("Episode text/timing/identity changed after Work lock")
    if strict and (spec.get("editorial_approved") is not True
                   or not isinstance(spec.get("approved_by"),str)
                   or len(spec["approved_by"])<3):
        raise ValueError("Work story not explicitly editorially approved by owner")
    return {"work_lock_valid":True,"script_sha256":expected,
            "editorial_approved":spec.get("editorial_approved") is True}

def main():
    p=argparse.ArgumentParser(description="Print fingerprint of a Work-created script (does NOT approve it)")
    p.add_argument("--episode",required=True,type=Path)
    args=p.parse_args()
    ep=json.loads(args.episode.read_text(encoding="utf-8"))
    print(json.dumps({"episode":ep.get("id"),"script_sha256":script_fingerprint(ep),
        "next_step":"Work/owner must add this hash to work_lock; only owner grants actual approval"}))

if __name__=="__main__":
    main()
