"""Canon-constrained episodic writer. Offline outline mode works without GPU.

Local Ollama mode can draft original scene dialogue, but only inside locked
narrative beats and body/knowledge constraints. Generated scripts are always
UNAPPROVED drafts and cannot reach the scheduled publication queue.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from studio.continuity import BODIES, check_episode
from studio.render import validate_episode

ROOT=Path(__file__).resolve().parent.parent
PLAN=ROOT/"story/season1_beats.json"
MEMORY=ROOT/"story/story_memory.json"
CANON=ROOT/"docs/STORY_CANON.md"
FIELDS=("heading","caption","voiceover","shot")
SCHEMA={"type":"object","properties":{"scenes":{"type":"array","items":{
    "type":"object","properties":{k:{"type":"string"} for k in ("id",)+FIELDS},
    "required":list(("id",)+FIELDS),"additionalProperties":False}}},
    "required":["scenes"],"additionalProperties":False}
BANNED_LINES=("you decide", "vote now", "choose a", "choose b", "like and subscribe",
              "world ends today", "the universe explodes")
MAX_LEN={"heading":48,"caption":140,"voiceover":160,"shot":600}
ID_PATTERN=re.compile(r"^E\d{2,4}$")


def read_json(path:Path)->dict:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise ValueError("Expected JSON object")
    return value


def validate_plan(plan:dict, memory:dict)->None:
    if not isinstance(plan.get("episodes"),list) or not plan["episodes"]:
        raise ValueError("Season plan is empty")
    seen=set()
    for ep in plan["episodes"]:
        eid=ep.get("id")
        if not isinstance(eid,str) or not ID_PATTERN.fullmatch(eid) or eid in seen:
            raise ValueError("Invalid or duplicate episode IDs in plan")
        seen.add(eid)
        if ep.get("continuity_start")!="post" or ep.get("continuity_end")!="post":
            raise ValueError("Initial season outline must preserve current exchange state")
        beats=ep.get("beats")
        if not isinstance(beats,list) or not 4<=len(beats)<=12:
            raise ValueError(f"{eid}: expected 4..12 narrative beats")
        for i,beat in enumerate(beats,1):
            if beat.get("id")!=f"S{i:02d}":
                raise ValueError("Narrative beat IDs must be ordered S01...")
            body=beat.get("body")
            if body not in BODIES:
                raise ValueError(f"{eid}/{beat['id']}: unknown body")
            if beat.get("palette") not in ("earth","noctaris"):
                raise ValueError("Invalid location")
            if beat.get("palette")=="earth" and not body.startswith("earth_"):
                raise ValueError("Body depicted in the wrong physical world")
            if beat.get("palette")=="noctaris" and not body.startswith("noctaris_"):
                raise ValueError("Body depicted in the wrong physical world")
            if not isinstance(beat.get("shot"),str) or len(beat["shot"])<30:
                raise ValueError("Scene needs a concrete visual setup")
            for key in ("heading","caption","voiceover"):
                if not isinstance(beat.get(key),str) or len(beat[key])>MAX_LEN[key]:
                    raise ValueError(f"{eid}/{beat['id']}: invalid {key}")
            if any(x not in memory.get("hard_limits",[]) for x in []):
                raise ValueError("Unrecognized canon constraint")


def create_prompt(episode:dict,memory:dict,canon:str)->str:
    """Model gets actual current continuity, open threads and immutable scene beats."""
    fixed={
        "episode":episode["id"],"title":episode["title"],"premise":episode["premise"],
        "beats":[{"id":x["id"],"shot":x["shot"],"body":x["body"],
                  "identity":BODIES[x["body"]]["post"],
                  "location":x["palette"],"known_facts":x.get("known_facts",[])}
                  for x in episode["beats"]]
    }
    return (
      "You are a professional episodic screenwriter for the ORIGINAL fictional series TWO WORLDS.\n"
      "Draft six (or exactly the given number of) short ENGLISH motion-comic scene texts.\n"
      "Obey the story bible and ALL fixed beats. Do NOT swap or transport bodies. "
      "A character understands only languages explicitly learned. "
      "The listed body and occupant are DISTINCT; dialogue personality matches occupant. "
      "Dialogue is emotionally specific, varied, short, naturally spoken. "
      "No spontaneous magical abilities, omniscient exposition, viewer choices, "
      "pop-culture references, unprepared resurrection or convenient rescues. "
      "Do not introduce events contradicting locked scene actions. "
      "Write a coherent cause-and-effect mini-arc and compact captions. "
      "Use one voiceover line per scene prefixed by the speaking character, or an empty string.\n\n"
      "CANON:\n"+canon[:16000]+"\n\n"
      "KNOWN HISTORY (development continuity, NOT published):\n"+json.dumps(memory,ensure_ascii=False)+"\n\n"
      "LOCKED SCENES (you may polish text, NOT change the scene's physical actions):\n"
      +json.dumps(fixed,ensure_ascii=False)+"\n\n"
      "Return ONLY JSON object with scenes. Each entry MUST have id,heading,caption,voiceover,shot. "
      "Each id must match in sequence and do not include new IDs. Limits: heading <=48, "
      "caption <=140, voiceover <=160, shot <=600 characters. The given shots may be "
      "polished but the underlying action and reveal must remain identical."
    )


def call_ollama(prompt:str, *, url:str, model:str, timeout=180)->dict:
    """Explicit opt-in to local Ollama. Uses structured output for parseability."""
    if not url.startswith(("http://","https://")):
        raise ValueError("Ollama endpoint requires http(s)")
    payload={"model":model,"prompt":prompt,"format":SCHEMA,"stream":False,
             "options":{"temperature":0.45,"seed":31127,"num_ctx":8192,"num_predict":1900}}
    req=urllib.request.Request(url.rstrip("/")+"/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            wire=response.read(3*1024*1024)
    except Exception as exc:
        raise RuntimeError("Local Ollama generation unavailable; configure reachable local model") from exc
    result=json.loads(wire)
    if result.get("done") is False or not isinstance(result.get("response"),str):
        raise ValueError("Ollama returned incomplete response")
    return json.loads(result["response"])


def validate_dialogue(candidate:dict, beats:list[dict])->list[dict]:
    if not isinstance(candidate,dict) or not isinstance(candidate.get("scenes"),list):
        raise ValueError("Writer returned no scenes")
    if len(candidate["scenes"])!=len(beats):
        raise ValueError("Writer changed the locked number of scenes")
    output=[]
    for item, beat in zip(candidate["scenes"],beats):
        if not isinstance(item,dict) or item.get("id")!=beat["id"]:
            raise ValueError("Writer re-ordered scene IDs")
        text={}
        for key in FIELDS:
            value=item.get(key)
            if not isinstance(value,str) or (key=="shot" and len(value)<25) or len(value)>MAX_LEN[key]:
                raise ValueError(f"{beat['id']}: invalid text field {key}")
            if key in ("heading","caption","voiceover"):
                if any(phrase in value.casefold() for phrase in BANNED_LINES):
                    raise ValueError("Writer generated prohibited engagement bait or plot device")
            text[key]=value
        # A human editor must still verify factual accuracy and body/identity.
        output.append({"id":beat["id"],**text})
    return output


def baseline_dialogue(beats:list[dict])->dict:
    return {"scenes":[{"id":b["id"],**{key:b[key] for key in FIELDS}} for b in beats]}


def draft(episode_id:str, engine="outline", *, model="qwen2.5:7b",
          endpoint="http://127.0.0.1:11434",generator=None)->dict:
    memory=read_json(MEMORY)
    plan=read_json(PLAN)
    validate_plan(plan,memory)
    candidate=next((ep for ep in plan["episodes"] if ep["id"]==episode_id),None)
    if candidate is None:
        raise ValueError(f"Episode {episode_id} not in season plan")
    if episode_id in memory.get("published_episodes",[]):
        raise ValueError("Cannot automatically rewrite a published episode")
    canon=CANON.read_text(encoding="utf-8")
    prompt=create_prompt(candidate,memory,canon)
    if engine=="outline":
        result=baseline_dialogue(candidate["beats"])
        engine_name="locked_outline_preview_not_ai"
    elif engine=="ollama":
        result=(generator or call_ollama)(prompt,url=endpoint,model=model)
        engine_name="local_ollama"
    else:
        raise ValueError("Unknown writing engine")
    lines=validate_dialogue(result,candidate["beats"])
    scenes=[]
    for beat, line in zip(candidate["beats"],lines):
        fixed={key:copy.deepcopy(beat[key]) for key in
               ("id","duration","palette","effect","art")}
        fixed.update(line)
        fixed["shot"] = beat["shot"]  # plot action stays locked; model writes dialogue, not events
        # IMPORTANT: model cannot change assigned physical body/occupant.
        body=beat["body"]
        fixed["continuity"]={"state":"post","body":body,
                             "occupant":BODIES[body]["post"]}
        fixed["known_facts"]=beat.get("known_facts",[])
        scenes.append(fixed)
    episode={"id":candidate["id"],"title":candidate["title"],
             "subtitle":candidate["premise"][:140],"fps":12,
             "continuity_start":"post","continuity_end":"post",
             "scenes":scenes,
             "editorial_status":"draft_unapproved",
             "creative_backend":engine_name,
             "source_outline_sha256":hashlib.sha256(json.dumps(candidate,sort_keys=True).encode()).hexdigest(),
             "canon_sha256":hashlib.sha256(canon.encode()).hexdigest(),
             "known_history_snapshot":memory["snapshot"]}
    check_episode(episode,strict=True)
    validate_episode(episode,ROOT/"assets")
    return episode


def main():
    p=argparse.ArgumentParser(description="Create unapproved canon-constrained episode JSON")
    p.add_argument("--episode",required=True,help="E03, E04 etc.")
    p.add_argument("--engine",choices=("outline","ollama"),default="outline")
    p.add_argument("--model",default=os.getenv("OLLAMA_MODEL","qwen2.5:7b"))
    p.add_argument("--endpoint",default=os.getenv("OLLAMA_URL","http://127.0.0.1:11434"))
    p.add_argument("--output-dir",type=Path,default=ROOT/"drafts")
    p.add_argument("--overwrite",action="store_true",help="Explicitly replace existing unapproved draft")
    args=p.parse_args()
    ep=draft(args.episode,args.engine,model=args.model,endpoint=args.endpoint)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/(args.episode+".json")
    if path.is_file() and not args.overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite for an unapproved draft")
    path.write_text(json.dumps(ep,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Created UNAPPROVED draft: {path}; no production release or publishing scheduled.")


if __name__=="__main__":
    main()
