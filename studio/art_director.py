"""Consistent-character art director for a user-operated local ComfyUI GPU.

Default: outputs *detailed, repeatable prompt packs*. If --backend comfyui is
explicitly requested, submits a customizable workflow to a reachable ComfyUI
instance and downloads PNGs. Never claims face consistency has been verified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from studio.continuity import BODIES, check_episode
from studio.render import validate_episode

ROOT=Path(__file__).resolve().parent.parent
STYLES=ROOT/"story/visual_bible.json"
WORKFLOW=ROOT/"templates/comfyui_sdxl_api.json"
SUPPORTED={"__PROMPT__","__NEGATIVE__","__WIDTH__","__HEIGHT__",
           "__SEED__","__CHECKPOINT__","__PREFIX__"}
MAX_ART_BYTES=12*1024*1024


def load_style(path:Path=STYLES)->dict:
    d=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d.get("characters"),dict) or not isinstance(d.get("locations"),dict):
        raise ValueError("Incomplete visual bible")
    pairs=[("earth_mira","noctaris_lyra"),("earth_eli","noctaris_ilan")]
    for a,b in pairs:
        if d["characters"][a]["identity_id"]!=d["characters"][b]["identity_id"]:
            raise ValueError(f"Double {a}/{b} is not visually paired")
        if d["characters"][a]["seed_anchor"]!=d["characters"][b]["seed_anchor"]:
            raise ValueError(f"Double {a}/{b} has incompatible identity anchors")
    return d


def get_subjects(scene:dict,style:dict)->list[str]:
    ids=scene.get("subjects")
    if ids is None:
        ref=scene.get("continuity",{})
        ids=[ref["body"]] if ref.get("body") else []
    if not isinstance(ids,list) or any(not isinstance(x,str) or x not in style["characters"] for x in ids):
        raise ValueError(f"{scene.get('id')}: invalid physical body references")
    if not ids:
        raise ValueError(f"{scene.get('id')}: specify at least one body for professional art")
    if len(set(ids))!=len(ids) or len(ids)>3:
        raise ValueError(f"{scene.get('id')}: duplicate or excessive subjects")
    return ids


def panel_prompt(scene:dict,style:dict,episode_id:str)->dict:
    bodies=get_subjects(scene,style)
    location=scene.get("palette","earth")
    if location not in style["locations"]:
        raise ValueError("Unknown illustrated location")
    details=[]
    references={}
    for body in bodies:
        ch=style["characters"][body]
        details.append(f'{ch["body_name"]}: {ch["gender"]}, age {ch["age"]}; '
                       f'fixed facial identity {ch["body"]}; signature {ch["distinctive"]}; '
                       f'wardrobe: {ch["clothing"]}.')
        if ch.get("reference_file"):
            path=(ROOT/"assets"/ch["reference_file"]).resolve()
            if (ROOT/"assets").resolve() not in path.parents or not path.is_file():
                raise ValueError(f"Missing or unsafe approved reference image for {body}")
            if path.suffix.lower() not in (".png",".jpg",".jpeg"):
                raise ValueError("Unsupported character reference format")
            references[body]={"file":str(path.relative_to(ROOT)),
                              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    shot=scene.get("shot")
    if not isinstance(shot,str) or len(shot)<15:
        raise ValueError(f"{scene.get('id')}: scene needs specific art direction")
    identity="; ".join(details)
    prompt=(
        style["visual_style"]+" "+style["composition"]+" "+
        style["locations"][location]+" SCENE: "+shot+" "+
        "ON-SCREEN BODIES: "+identity+" "+
        "Character appearance follows the PHYSICAL BODY, not its consciousness. "+
        "Dialogue/text added later in editing; image must contain no legible words. "+
        "Readable cinematic posing, correctly articulated anatomy, consistent attire."
    )
    seed_anchor=sum(style["characters"][bid]["seed_anchor"] for bid in bodies)
    shot_hash=int(hashlib.sha256((episode_id+"/"+scene["id"]).encode()).hexdigest()[:8],16)
    seed=(seed_anchor+shot_hash)%2147483647
    return {"episode":episode_id,"scene":scene["id"],"bodies":bodies,
            "location":location,"seed":seed,"prompt":prompt,
            "negative":style["negative"],"references":references,
            "review_status":"unreviewed","consistency_verified":False,
            "note":"Same descriptions/seeds support repeatability, but DO NOT guarantee identical faces. "
                   "Use approved multi-angle references with a ComfyUI IP-Adapter custom workflow."}


def compile_prompt_pack(episode:dict,style:dict)->dict:
    check_episode(episode,strict=True)
    validate_episode(episode,ROOT/"assets")
    entries=[panel_prompt(scene,style,episode["id"]) for scene in episode["scenes"]]
    return {"schema":1,"episode":episode["id"],"backend":"unexecuted",
            "reference_guidance":"Use approved reference sheets through IP-Adapter/ControlNet workflow "
                                 "for persistent faces and poses, then manually inspect all generated panels.",
            "scenes":entries}


def substitute(obj, values:dict):
    """Replace typed exact placeholders without evaluating arbitrary strings."""
    if isinstance(obj,dict):
        return {k:substitute(v,values) for k,v in obj.items()}
    if isinstance(obj,list):
        return [substitute(x,values) for x in obj]
    if isinstance(obj,str) and obj in values:
        return values[obj]
    return obj


def comfy_workflow(prompt:dict, template:dict, checkpoint:str, width:int=768, height:int=1344)->dict:
    if not checkpoint or checkpoint.strip()!=checkpoint:
        raise ValueError("Set a checkpoint name installed on your ComfyUI instance")
    if width%64 or height%64 or not (512<=width<=2048 and 512<=height<=2048):
        raise ValueError("ComfyUI dimensions must be 512..2048 in multiples of 64")
    replacements={
        "__PROMPT__":prompt["prompt"], "__NEGATIVE__":prompt["negative"],
        "__SEED__":prompt["seed"], "__CHECKPOINT__":checkpoint,
        "__WIDTH__":width, "__HEIGHT__":height,
        "__PREFIX__":f'TWO_WORLDS_{prompt["episode"]}_{prompt["scene"]}'
    }
    workflow=substitute(template,replacements)
    if not isinstance(workflow,dict) or "9" not in workflow:
        raise ValueError("Expected ComfyUI API JSON workflow with SaveImage node 9")
    flat=json.dumps(workflow)
    unresolved=[x for x in SUPPORTED if x in flat]
    if unresolved:
        raise ValueError("Unresolved workflow placeholders: "+", ".join(unresolved))
    return workflow


def _http_json(url:str,payload=None,timeout=60):
    body=json.dumps(payload).encode("utf-8") if payload is not None else None
    req=urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"},method="POST" if body else "GET")
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.loads(response.read(4*1024*1024).decode("utf-8"))


def comfy_render(workflow:dict,base_url:str,out:Path,timeout=240,client=None)->dict:
    """Interact with user's ComfyUI HTTP server; fails closed on missing jobs/images."""
    from PIL import Image
    if not base_url.startswith(("http://","https://")):
        raise ValueError("ComfyUI URL must be http(s)")
    api=base_url.rstrip("/")
    post=(client or _http_json)(api+"/prompt",{"prompt":workflow})
    if "error" in post:
        raise RuntimeError("ComfyUI prompt rejected: "+str(post["error"]))
    pid=post.get("prompt_id")
    if not pid or not isinstance(pid,str):
        raise RuntimeError("ComfyUI did not return prompt_id")
    deadline=time.monotonic()+timeout
    completed=None
    while time.monotonic()<deadline:
        history=(client or _http_json)(api+"/history/"+urllib.parse.quote(pid,safe=""))
        if pid in history:
            completed=history[pid]
            break
        time.sleep(.8)
    if not completed:
        raise TimeoutError("ComfyUI job did not complete by deadline")
    status=completed.get("status",{})
    if status.get("status_str")=="error" or status.get("completed") is False:
        raise RuntimeError("ComfyUI job failed; inspect ComfyUI logs")
    outputs=completed.get("outputs",{})
    images=outputs.get("9",{}).get("images",[])
    if not images:
        raise RuntimeError("SaveImage node 9 produced no image")
    img=images[0]
    for key in ("filename","subfolder","type"):
        if not isinstance(img.get(key),str):
            raise RuntimeError("Malformed ComfyUI image reference")
    qs=urllib.parse.urlencode({"filename":img["filename"],"subfolder":img["subfolder"],"type":img["type"]})
    req=urllib.request.Request(api+"/view?"+qs)
    with urllib.request.urlopen(req,timeout=60) as resp:
        data=resp.read(MAX_ART_BYTES+1)
    if len(data)>MAX_ART_BYTES:
        raise RuntimeError("ComfyUI image exceeds allowed size")
    out.parent.mkdir(parents=True,exist_ok=True)
    # Verify payload and dimensions before writing.
    import io
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        if image.format!="PNG" or min(image.size)<512:
            raise RuntimeError("ComfyUI returned unexpected format or tiny image")
    out.write_bytes(data)
    return {"file":str(out),"sha256":hashlib.sha256(data).hexdigest(),
            "comfy_prompt_id":pid,"review_status":"unreviewed",
            "consistency_verified":False}


def run(episode:Path,output:Path,backend="prompts",workflow_file:Path=WORKFLOW,
        checkpoint=None,base_url=None,width=768,height=1344):
    style=load_style()
    ep=json.loads(episode.read_text(encoding="utf-8"))
    pack=compile_prompt_pack(ep,style)
    if backend not in ("prompts","comfyui"):
        raise ValueError("Unknown art backend")
    out=output/ep["id"]
    out.mkdir(parents=True,exist_ok=True)
    if backend=="comfyui":
        if checkpoint is None:
            checkpoint=os.environ.get("COMFYUI_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("COMFYUI_CHECKPOINT is required for actual image generation")
        workflow=json.loads(workflow_file.read_text(encoding="utf-8"))
        for scene in pack["scenes"]:
            if len(scene["bodies"])>1 and len(scene["references"])>1:
                # Composite multi-character panels need an explicit multi-reference workflow.
                raise ValueError("Multi-character reference control needs a custom ComfyUI workflow and art review")
            graph=comfy_workflow(scene,workflow,checkpoint,width,height)
            img=out/(scene["scene"]+".png")
            result=comfy_render(graph,base_url or os.environ.get("COMFYUI_URL","http://127.0.0.1:8188"),img)
            scene["generated"]=result
        pack["backend"]="comfyui"
    dest=out/"prompts.json"
    dest.write_text(json.dumps(pack,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return dest


def main():
    p=argparse.ArgumentParser(description="Consistent-identity art prompt packs or optional ComfyUI render")
    p.add_argument("--episode",type=Path,default=ROOT/"episodes/E01_THE_REFLECTION.json")
    p.add_argument("--output-dir",type=Path,default=ROOT/"dist/art")
    p.add_argument("--backend",choices=("prompts","comfyui"),default="prompts")
    p.add_argument("--workflow",type=Path,default=WORKFLOW)
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--url",default=None)
    p.add_argument("--width",type=int,default=768)
    p.add_argument("--height",type=int,default=1344)
    args=p.parse_args()
    print(run(args.episode,args.output_dir,args.backend,args.workflow,
              args.checkpoint,args.url,args.width,args.height))


if __name__=="__main__":
    main()
