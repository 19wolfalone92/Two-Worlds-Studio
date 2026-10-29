"""New character designs from text, then consistent views from their OWN output.

No Work artwork is needed or silently reused. This factory keeps new designs
separate from canonical references. A drawing backend must actually execute a
job; planning never counts as generated artwork or artistic approval.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from PIL import Image

from studio.character_art import (VIEWS, _id, digest, fingerprint, inspect_image,
                                  read_json, resolve_asset, write_json)

SCHEMA = 1
TASKS = ("front", "three_quarter", "profile", "back", "expressions", "blink", "hands")


def _validate_spec(spec: dict) -> None:
    if spec.get("schema") != SCHEMA or not isinstance(spec.get("style"), str) or len(spec["style"].strip()) < 20:
        raise ValueError("Provide schema=1 and an explicit new drawing style")
    bodies = spec.get("bodies")
    if not isinstance(bodies, dict) or not 1 <= len(bodies) <= 6:
        raise ValueError("Design 1..6 explicit physical bodies at a time")
    for name, body in bodies.items():
        _id(name)
        if not isinstance(body, dict):
            raise ValueError("Body specification must be an object")
        if any(k in body for k in ("references", "reference_file", "reference_images")):
            raise ValueError("From-scratch specification must not contain existing image references")
        _id(body.get("identity_version"))
        _id(body.get("identity_group", name))
        if not isinstance(body.get("description"), str) or len(body["description"].strip()) < 30:
            raise ValueError("Explicit new physical appearance description required")
        if not isinstance(body.get("anchors"), dict) or not body["anchors"] or any(
                not isinstance(v, str) or not v.strip() for v in body["anchors"].values()):
            raise ValueError("Provide named appearance/costume anchors")
        if not isinstance(body.get("avoid", []), list) or any(not isinstance(v,str) for v in body.get("avoid",[])):
            raise ValueError("avoid must contain drawing instructions")


def create_design(spec_path: Path, output: Path) -> dict:
    spec = read_json(spec_path)
    _validate_spec(spec)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new design directory; preserve previous character versions")
    project = {"schema": SCHEMA, "kind": "studio_from_scratch_design",
               "origin": "text_only_no_input_artwork", "spec": spec,
               "body_order":list(spec["bodies"]),
               "source_spec_sha256": digest(spec_path),
               "canonical_replacement": False, "publication_approved": False}
    project["design_fingerprint"] = fingerprint(project)
    write_json(output/"design.json", project)
    write_json(output/"ledger.json", {"schema": SCHEMA, "design_fingerprint": project["design_fingerprint"], "results": []})
    return project


def load_design(root: Path) -> dict:
    project = read_json(root/"design.json")
    saved = project.pop("design_fingerprint", None)
    if project.get("kind") != "studio_from_scratch_design" or fingerprint(project) != saved:
        raise ValueError("Character design changed; create a new design version")
    _validate_spec(project["spec"])
    if set(project.get("body_order",[]))!=set(project["spec"]["bodies"]) or len(project["body_order"])!=len(project["spec"]["bodies"]):
        raise ValueError("Invalid explicit physical-body design order")
    project["design_fingerprint"] = saved
    return project


def load_results(root: Path) -> list[dict]:
    project = load_design(root)
    ledger = read_json(root/"ledger.json")
    if ledger.get("design_fingerprint") != project["design_fingerprint"]:
        raise ValueError("Result ledger belongs to another character design")
    results = ledger.get("results")
    if not isinstance(results, list):
        raise ValueError("Invalid generation ledger")
    seen = set()
    for item in results:
        _id(item.get("job_id"))
        if item.get("job_id") in seen:
            raise ValueError("Duplicate completed generation job")
        seen.add(item.get("job_id"))
        body=project["spec"]["bodies"].get(item.get("body_id"))
        if (not body or item.get("view") not in TASKS or
            item["identity_version"]!=body["identity_version"] or
            item["job_id"]!=f"{item['body_id']}__{item['view']}"):
            raise ValueError("Generation result has a different physical body, view or identity version")
        receipt=read_json(root/f"receipts/{item['job_id']}.json")
        job=copy.deepcopy(receipt.get("job",{}))
        saved_job=job.pop("job_fingerprint",None)
        if (receipt.get("result")!=item or fingerprint(job)!=saved_job or
            saved_job!=item.get("job_fingerprint") or
            job.get("design_fingerprint")!=project["design_fingerprint"] or
            job.get("references")!=item.get("consumed_references")):
            raise ValueError("Generation ledger and exact execution receipt disagree")
        path = resolve_asset(root, item["file"])
        if digest(path) != item["sha256"]:
            raise ValueError("Generated artwork changed after registration")
        for ref in item.get("consumed_references", []):
            if digest(resolve_asset(root, ref["file"])) != ref["sha256"]:
                raise ValueError("Generated identity reference changed")
    return results


def _groups(project: dict) -> dict[str, str]:
    # Order is part of the immutable design contract. The first body in each
    # group creates the face; doubles receive that new face, their own costume.
    result = {}
    for name in project["body_order"]:
        body = project["spec"]["bodies"][name]
        result.setdefault(body.get("identity_group", name), name)
    return result


def ready_jobs(root: Path, views: tuple[str, ...] = TASKS) -> list[dict]:
    if not views or len(set(views)) != len(views) or any(v not in TASKS for v in views):
        raise ValueError("Choose distinct supported drawing tasks")
    project = load_design(root)
    results = load_results(root)
    indexed = {(r["body_id"],r["view"]): r for r in results}
    groups = _groups(project)
    jobs = []
    # First establish all bodies before allocating generation to other views.
    for view in ("front",) + tuple(v for v in views if v != "front"):
        for name in project["body_order"]:
            body = project["spec"]["bodies"][name]
            if (name,view) in indexed:
                continue
            primary = groups[body.get("identity_group",name)]
            own_base = indexed.get((name,"front"))
            group_base = indexed.get((primary,"front"))
            refs = []
            if view != "front":
                if own_base is None:
                    continue
                source = own_base
                stage = "own_generated_identity_variant"
            elif primary != name:
                if group_base is None:
                    continue
                source = group_base
                stage = "shared_face_new_body_design"
            else:
                source = None
                stage = "text_only_bootstrap"
            if source:
                refs = [{"file":source["file"], "sha256":source["sha256"],
                         "body_id":source["body_id"], "role":"newly_generated_face_and_identity",
                         "review_status":"unreviewed"}]
            prompt = ("Use case: stylized-concept. TWO WORLDS Studio NEW CHARACTER DESIGN.\n"
                      "Style: " + project["spec"]["style"] + "\n"
                      "PHYSICAL BODY " + name + ": " + body["description"] + "\n"
                      "FIXED ANCHORS: " + "; ".join(k+": "+v for k,v in body["anchors"].items()) + ".\n"
                      "DRAW THIS: " + VIEWS[view] + "\n"
                      "Do not imitate any franchise or an existing celebrity. No text, watermark, "
                      "logos or extra characters. Correct joints, fingers and eye anatomy. "
                      "No horizontal flipping: anatomical side markers remain attached to their body. ")
            if not refs:
                prompt += "Invent this character's original facial design FROM TEXT ALONE; there are NO input images. "
            elif stage == "shared_face_new_body_design":
                prompt += ("The reference is a NEWLY GENERATED physical double. Retain her exact facial "
                           "bone structure, eye shape and age; draw this body's distinct hair and costume "
                           "from the target description. Do not copy the other body's outfit. ")
            else:
                prompt += "Use ONLY this body's own newly generated reference; retain the same face, age, hairstyle and costume. "
            if body.get("avoid"):
                prompt += "AVOID: " + "; ".join(body["avoid"]) + ". "
            sheet = view in {"expressions","blink","hands"}
            if not sheet:
                prompt += "One full figure inside the canvas, feet visible, 5% margin, neutral matte studio background, no aura."
            job = {"schema":SCHEMA, "id":f"{name}__{view}", "body_id":name, "view":view,
                   "identity_version":body["identity_version"], "stage":stage,
                   "design_fingerprint":project["design_fingerprint"], "prompt":prompt,
                   "references":refs, "target_pixels":[1536,1024] if sheet else [1024,1536],
                   "seed":int(fingerprint({"design":project["design_fingerprint"],"body":name,"view":view})[:8],16),
                   "review_status":"unreviewed", "publication_approved":False, "rig_ready":False}
            job["job_fingerprint"] = fingerprint(job)
            jobs.append(job)
    return jobs


def export_job(root: Path, job_id: str, output: Path | None = None) -> dict:
    _id(job_id)
    jobs = [j for j in ready_jobs(root) if j["id"] == job_id]
    if len(jobs) != 1:
        raise ValueError("Job is completed, unknown, or blocked waiting for its newly generated identity")
    if output:
        write_json(output, jobs[0])
    return jobs[0]


def register_result(root: Path, job: dict, image: Path, *, backend: str,
                    execution: dict) -> dict:
    project = load_design(root)
    old_results = load_results(root)
    match = [r for r in old_results if r["job_id"] == job.get("id")]
    if match:
        if len(match) == 1 and match[0]["sha256"] == digest(image) and match[0]["job_fingerprint"] == job.get("job_fingerprint"):
            return match[0]
        raise FileExistsError("Completed drawing is immutable; create a new design version")
    expected = export_job(root, job.get("id"))
    if expected != job:
        raise ValueError("Generation job or identity inputs differ from the ready factory task")
    if not backend.strip() or execution.get("prompt_sha256") != fingerprint({"prompt":job["prompt"]}):
        raise ValueError("Record the actual backend and exact executed prompt fingerprint")
    if execution.get("reference_sha256") != [r["sha256"] for r in job["references"]]:
        raise ValueError("Actual consumed references do not match this generation job")
    stats = inspect_image(image)
    with Image.open(image) as im:
        im.load()
        if im.format != "PNG":
            raise ValueError("Expected the original generated PNG")
    if max(stats["width"],stats["height"]) < 768 or stats["contrast"] < .015:
        raise ValueError("Generated image is too small or almost blank; do not mark this job complete")
    relative = f"images/{job['body_id']}/{job['view']}_{stats['sha256'][:16]}.png"
    target = root/relative
    if target.exists():
        raise FileExistsError("Unregistered output already exists; inspect the incomplete import")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(image.read_bytes())
    result = {"schema":SCHEMA, "job_id":job["id"], "body_id":job["body_id"],
              "view":job["view"], "identity_version":job["identity_version"],
              "file":relative, "sha256":stats["sha256"], "job_fingerprint":job["job_fingerprint"],
              "stage":job["stage"], "consumed_references":copy.deepcopy(job["references"]),
              "backend":backend, "execution":copy.deepcopy(execution), "technical":stats,
              "human_review":None, "identity_automatically_verified":False,
              "publication_approved":False, "rig_ready":False}
    write_json(root/f"receipts/{job['id']}.json", {"job":job, "result":result})
    write_json(root/"ledger.json", {"schema":SCHEMA,"design_fingerprint":project["design_fingerprint"],
                                   "results":old_results+[result]})
    return result


def run_batch(root: Path, backend, *, max_images: int = 4,
              views: tuple[str,...] = TASKS) -> dict:
    """Execute dependency-ordered jobs with one already-created model session.

    backend(job, output) must generate actual bytes and return provenance. A
    test fake or host bridge is always labelled by its supplied backend name.
    """
    if type(max_images) is not int or not 1 <= max_images <= 24:
        raise ValueError("Use an explicit batch limit from 1 to 24 images")
    before = len(load_results(root))
    for _ in range(max_images):
        jobs = ready_jobs(root,views)
        if not jobs:
            break
        job = jobs[0]
        output = root/f"working/{job['id']}.png"
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            raise FileExistsError("An incomplete attempt exists; inspect it before retrying")
        info = backend(job,output)
        if not output.is_file():
            raise RuntimeError("Backend returned without creating an actual drawing")
        register_result(root,job,output,backend=info["backend"],execution=info["execution"])
        output.unlink()
    return {"generated_this_run":len(load_results(root))-before, **status(root)}


class FluxBackend:
    """One lazily loaded GPU model shared by all sequential drawing jobs."""
    def __init__(self, root: Path, *, width: int = 768, height: int = 1152, steps: int = 4):
        self.root,self.width,self.height,self.steps = root,width,height,steps
        self.session = None

    def __call__(self, job: dict, output: Path) -> dict:
        from studio.kaggle_flux import GPUImageSession, validate_reference, validate_gpu_request
        ref = job["references"][0] if job["references"] else None
        payload = {"prompt":job["prompt"], "seed":job["seed"],
                   "reference":validate_reference(resolve_asset(self.root,ref["file"])) if ref else None}
        w,h = (self.height,self.width) if job["view"] in {"expressions","blink","hands"} else (self.width,self.height)
        validate_gpu_request(payload,output,width=w,height=h,steps=self.steps)
        if self.session is None:
            self.session = GPUImageSession()
        result = self.session.render(payload,output,width=w,height=h,steps=self.steps)
        return {"backend":"studio_flux_gpu", "execution":{
                "prompt_sha256":fingerprint({"prompt":job["prompt"]}),
                "reference_sha256":[r["sha256"] for r in job["references"]],
                "model":result["model"],"gpu_name":result["gpu_name"],"precision":result["precision"],
                "offload_mode":result.get("offload_mode"),"gpu_total_gib":result.get("gpu_total_gib"),
                "worker":"studio.character_factory.FluxBackend"}}


def status(root: Path) -> dict:
    project, results = load_design(root), load_results(root)
    primary = set(_groups(project).values())
    return {"design_origin":project["origin"], "actual_drawings":len(results),
            "text_only_bootstrap_drawings":sum(r["stage"]=="text_only_bootstrap" for r in results),
            "external_input_images":0,
            "physical_bodies_with_front":[r["body_id"] for r in results if r["view"]=="front"],
            "initial_designs_required":len(primary),
            "backends":sorted({r["backend"] for r in results}),
            "publication_approved":False,"rig_ready":False,"canonical_replacement":False}


def export_art_project(root: Path, output: Path) -> Path:
    from studio.character_art import create_project, build_jobs
    project, results = load_design(root), load_results(root)
    spec = copy.deepcopy(project["spec"])
    if any(not any(r["body_id"]==b and r["view"]=="front" for r in results) for b in spec["bodies"]):
        raise ValueError("Every new physical body needs a real front drawing before export")
    staging = root/"art_project_source"
    if staging.exists():
        raise FileExistsError("Reference export already exists; preserve this version")
    registry = {"schema":1,"bodies":{}}
    for body_id, body in spec["bodies"].items():
        refs = []
        for result in results:
            if result["body_id"]!=body_id or result["view"] not in {"front","profile","three_quarter","back"}:
                continue
            source = resolve_asset(root,result["file"])
            # Body namespaces avoid collisions between equal pose names.
            rel = f"identities/{body_id}/{source.name}"
            target = staging/"assets"/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
            refs.append({"file":rel,"view":result["view"],"sha256":result["sha256"],
                         "review_status":"unreviewed","rights":"original",
                         "rights_note":"New AI-generated candidate from the Studio from-scratch drawing factory; human review pending."})
        registry["bodies"][body_id] = {"version":body["identity_version"],"references":refs}
    write_json(staging/"story/identity_registry.json",registry)
    write_json(staging/"appearance.json",spec)
    path = create_project(staging,staging/"appearance.json",output)
    build_jobs(output)
    return path


def main():
    parser = argparse.ArgumentParser(description="Create new character art from text, then reuse only its own generated identities")
    sub = parser.add_subparsers(dest="command",required=True)
    create = sub.add_parser("create")
    create.add_argument("--spec",type=Path,required=True)
    create.add_argument("--output",type=Path,required=True)
    for name in ("next","register","run-gpu","status","export"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--project",type=Path,required=True)
        if name in {"next","register"}:
            cmd.add_argument("--job",required=True)
        if name in {"next","export"}:
            cmd.add_argument("--output",type=Path,required=True)
        if name=="register":
            cmd.add_argument("--job-file",type=Path,required=True)
            cmd.add_argument("--image",type=Path,required=True)
            cmd.add_argument("--backend",required=True)
            cmd.add_argument("--execution",type=Path,required=True)
        if name=="run-gpu":
            cmd.add_argument("--max-images",type=int,default=4)
            cmd.add_argument("--views",default=",".join(TASKS))
            cmd.add_argument("--width",type=int,default=768)
            cmd.add_argument("--height",type=int,default=1152)
    a = parser.parse_args()
    if a.command=="create":
        result = create_design(a.spec,a.output)
    elif a.command=="next":
        result = export_job(a.project,a.job,a.output)
    elif a.command=="register":
        job = read_json(a.job_file)
        if job.get("id")!=a.job:
            raise ValueError("Job ID and supplied job file disagree")
        result = register_result(a.project,job,a.image,backend=a.backend,execution=read_json(a.execution))
    elif a.command=="run-gpu":
        result = run_batch(a.project,FluxBackend(a.project,width=a.width,height=a.height),
                           max_images=a.max_images,views=tuple(a.views.split(",")))
    elif a.command=="export":
        result = {"art_project":str(export_art_project(a.project,a.output))}
    else:
        result = status(a.project)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
