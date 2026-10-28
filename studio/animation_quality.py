"""Visual-performance production gate: actively prevents sticker-only masters.

Only checks explicit artistic signoff metadata; it does not pretend automatic
machine vision has proven photorealistic contact shadows or convincing acting.
"""
from __future__ import annotations
import json
from pathlib import Path

KINDS={"rigged_2d","drawn_sequence","generated_motion","intentional_hold"}
CUES={"pose","gaze","expression","head_turn","hand_action","walking",
      "body_action","cloth_motion","interaction"}

def audit_performance(episode:dict,*,publish:bool=False)->dict:
    out=[]
    for scene in episode.get("scenes",[]):
        actors=[x for x in scene.get("cinematic",{}).get("layers",[])
                if x.get("role")=="character"]
        if not actors:
            continue
        anim=scene.get("animation",{})
        issues=[]
        kind=anim.get("kind")
        if kind not in KINDS:
            issues.append("Missing authored actor-performance format (not just camera pan/zoom)")
        cues=anim.get("cues",[])
        if not isinstance(cues,list):
            issues.append("Animation cues must be a list")
        else:
            if kind!="intentional_hold" and not any(c in CUES for c in cues):
                issues.append("No expressive pose/gaze/gesture/motion cues declared")
        grounded=anim.get("integration",{})
        if not isinstance(grounded,dict) or not all(
                grounded.get(k) is True for k in ("contact_shadow","lighting","perspective")):
            issues.append("Unreviewed contact shadow, environmental lighting or perspective match")
        if anim.get("blink_style")=="overlay_only":
            issues.append("Unsafe sticker-like blink overlay is banned")
        if kind=="intentional_hold" and len(str(anim.get("hold_reason","")).strip())<15:
            issues.append("Stationary acting needs documented dramatic reason")
        if anim.get("reviewed_by") is None or len(str(anim["reviewed_by"]).strip())<3:
            issues.append("Human shot-in-environment visual review missing")
        out.append({"scene":scene.get("id"),"character_layers":len(actors),
                    "kind":kind,"issues":issues,"performance_pass":not issues})
    issues=sum(len(s["issues"]) for s in out)
    if publish and issues:
        failure=next(s for s in out if s["issues"])
        raise ValueError(f"{failure['scene']}: cinematic performance not master-ready: "
                         +"; ".join(failure["issues"]))
    return {"scenes_with_actors":len(out),
            "problems":issues,"shots":out,
            "master_ready":issues==0 and bool(out),
            "automatic_aesthetic_certification":False}
