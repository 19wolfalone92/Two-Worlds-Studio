"""Work-to-studio visual director: create shot JOBS, never scenes or dialogue.

Maps Work-authored physical scenes into camera/layer/lighting instructions
that humans can override before production. Does not fabricate story beats.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from studio.continuity import check_episode
from studio.work_contract import script_fingerprint

ROOT=Path(__file__).resolve().parent.parent
CAMERAS={
 "establishing":{"start":{"x":-.2,"y":.01,"zoom":1.06},
                 "end":{"x":.17,"y":-.03,"zoom":1.15}},
 "portrait":{"start":{"x":0,"y":0,"zoom":1.04},
             "end":{"x":.05,"y":-.08,"zoom":1.13}},
 "action":{"start":{"x":-.10,"y":.04,"zoom":1.12},
           "end":{"x":.12,"y":-.06,"zoom":1.23}},
 "mystery":{"start":{"x":-.10,"y":0,"zoom":1.03},
            "end":{"x":.06,"y":.01,"zoom":1.18}}
}
LIGHTING={
 "earth":{"key":"cool window/rain diffuse source","fill":"neutral blue reflected street light",
          "rim":"subtle warm practical highlights"},
 "noctaris":{"key":"motivated moon / torch keylight","fill":"deep copper-violet ambient bounce",
             "rim":"silver atmospheric backlight"}
}
REQUIRED_LAYERS=[
 {"role":"background","depth":-.85,"filename":"background.png"},
 {"role":"midground","depth":-.15,"filename":"midground_rgba.png"},
 {"role":"character","depth":.45,"filename":"character_rgba.png"},
 {"role":"foreground","depth":.85,"filename":"foreground_rgba.png"}
]

def classify(scene:dict)->str:
    """Composition heuristic from shot directions, NEVER determines plot."""
    shot=str(scene.get("shot","")).lower()
    if any(x in shot for x in ("army","siege","fighting","battle","runs","chase")):
        return "action"
    if any(x in shot for x in ("close-up","closeup","portrait","face","tears","looks at")):
        return "portrait"
    if any(x in shot for x in ("castle","landscape","city","skyline","fortress","two moons")):
        return "establishing"
    return "mystery"

def planned_shots(episode:dict, *, require_work_lock:bool=False)->dict:
    check_episode(episode,strict=True)
    if require_work_lock:
        from studio.work_contract import verify_work_lock
        verify_work_lock(episode,strict=True)
    eid=str(episode.get("id",""))
    if not eid or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in eid):
        raise ValueError("Unsafe episode ID")
    outputs=[]
    time_cursor=0.0
    for scene in episode["scenes"]:
        sid=scene["id"]
        if not isinstance(sid,str) or not sid.startswith("S") or not sid[1:].isdigit():
            raise ValueError("Unexpected Work scene ID")
        shot=classify(scene)
        world=scene.get("palette")
        if world not in LIGHTING:
            raise ValueError(f"{sid}: unknown world")
        expected=[
          {"role":item["role"],"path":f"{eid}/{sid}/{item['filename']}",
           "depth":item["depth"],"review_status":"unreviewed"}
          for item in REQUIRED_LAYERS
        ]
        center=scene.get("continuity",{})
        body=center.get("body")
        identity=scene.get("subjects") or ([body] if body else [])
        outputs.append({
          "scene":sid,"time_start":round(time_cursor,3),"duration":scene["duration"],
          "shot_preset":shot,"camera":CAMERAS[shot],
          "lighting":LIGHTING[world],"physical_bodies":identity,
          "actor_consciousness":center.get("occupant"),
          "required_illustration_layers":expected,
          "safe_zone":{"caption_top":.59,"caption_bottom":.85,
                       "right_ui_reserved":.14,"bottom_ui_reserved":.15},
          "review_status":"draft_visual_direction",
          "manual_director_override_allowed":True
        })
        time_cursor+=float(scene["duration"])
    return {"schema":1,"episode":eid,"source":"work_authored_locked_scenes_only",
            "script_sha256":script_fingerprint(episode),
            "seconds":round(time_cursor,3),"shots":outputs,
            "warning":"These are composition suggestions, not new narrative content or approved art."}

def main():
    p=argparse.ArgumentParser(description="Create a Work-script-to-art job list; no scripts created")
    p.add_argument("--episode",required=True,type=Path)
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--approved-work-only",action="store_true")
    args=p.parse_args()
    ep=json.loads(args.episode.read_text(encoding="utf-8"))
    job=planned_shots(ep,require_work_lock=args.approved_work_only)
    data=json.dumps(job,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(data,encoding="utf-8")
    print(data)

if __name__=="__main__":
    main()
