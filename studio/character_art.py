"""Reference-first drawing desk for the existing Studio.

Private Work art -> versioned character project -> reference-conditioned jobs ->
immutable candidates -> side-by-side review. No story authoring, face-recognition
claim, automatic approval, model download or public upload is performed here.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path

from PIL import Image

from studio.premium_assets import resolve_asset, inspect_image

SAFE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,95}\Z")
VIEWS = {
    "front": "One complete standing FRONT view, including feet, neutral relaxed hands, level camera.",
    "three_quarter": "One complete standing THREE-QUARTER view, including feet, same bone structure and wardrobe.",
    "profile": "One complete standing strict SIDE PROFILE, including feet, anatomically correct nose and ear.",
    "back": "One complete standing BACK view, back of head and clothing visible, face NOT visible, including feet.",
    "rear_three_quarter": "One complete standing REAR THREE-QUARTER view, facing away from camera, including feet.",
    "expressions": "Six equal head-and-shoulder panels in a 3-column, 2-row sheet: neutral, concern, restrained fear, resolve, sadness, subtle relief. Exactly the same character, head scale, camera angle, hairstyle, outfit and lighting in every panel; change facial acting only.",
    "blink": "Six equal close portrait panels in a 3-column, 2-row sheet: eyes open, eyelids descending halfway, eyes fully closed, short closed hold, reopening halfway, eyes open. Same stationary head, gaze, expression, lighting and scale in every panel. Closed lids REPLACE the open eyes completely. No eyeball drawn over a lid, no extra lash or double eye. This is a drawn reference sheet, NOT an aligned animation rig.",
    "hands": "Six equal panels in a 3-column, 2-row sheet showing this character's hands and costume cuffs: relaxed palm, relaxed back of hand, open palm toward camera, palm side view, gentle grip, fingers touching a flat glass plane. Anatomically correct fingers, knuckles, tendons and thumb attachment; hands belong to this same body.",
    "rig_parts": "A separated RGBA puppet-parts plate: head with back hair, torso with hidden shoulder/hip overlaps, left/right upper arms, forearms, hands, upper legs, lower legs and feet. Draw complete joint overlaps, consistent scale and the requested BACK view. Place pieces in separate non-overlapping cells; TRUE transparent background. This plate still requires artist alignment, pivots, hierarchy, occlusion and motion tracks before use in CharacterRig. Never call a plate a finished rig.",
}
REFERENCE_POSE = {
    "three_quarter": (
        "Turn the reference character's ENTIRE head, shoulders, hips and feet 45 degrees toward image-left. "
        "Draw a clearly diagonal THREE-QUARTER view: near shoulder broader, far shoulder foreshortened, "
        "nose pointing diagonally left, far cheek narrower, both eyes visible. "
        "Replace the frontal pose; do not merely recolour or trace the reference."
    ),
    "profile": (
        "Turn the whole character 90 degrees toward image-left into strict SIDE PROFILE. "
        "Head and body face the same side; show one eye, the nose silhouette and overlapping shoulders."
    ),
    "back": (
        "Turn the whole character 180 degrees, facing completely AWAY from the camera. "
        "Show the back of the head, torso, hands and feet; hide the entire face."
    ),
}

CRITERIA = ("identity", "anatomy", "asymmetry", "wardrobe", "pose", "eyes", "style")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        staged = Path(tmp.name)
    staged.replace(path)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _id(value: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError("Invalid character/job identifier")
    return value


def create_project(handoff: Path, spec_path: Path, output: Path) -> Path:
    """Import exact private reference bytes, preserving their review/provenance.

    Appearance is explicitly authored from the handoff. Public fixture prose
    cannot silently override the current private character design.
    """
    registry_path = handoff / "story/identity_registry.json"
    registry, spec = read_json(registry_path), read_json(spec_path)
    if registry.get("schema") != 1 or spec.get("schema") != 1:
        raise ValueError("Reference registry and appearance spec need schema=1")
    if not isinstance(spec.get("style"), str) or len(spec["style"].strip()) < 20:
        raise ValueError("Provide an explicit drawing style from current Work art")
    bodies = spec.get("bodies")
    if not isinstance(bodies, dict) or not bodies:
        raise ValueError("Provide explicit physical-body appearance specifications")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new project directory; existing art is preserved")
    staged_bodies, copies = {}, []
    for body_id, appearance in bodies.items():
        _id(body_id)
        source = registry.get("bodies", {}).get(body_id)
        if not source or not source.get("references"):
            raise ValueError(f"{body_id}: no actual Work reference art; cannot invent an identity")
        if appearance.get("identity_version") != source.get("version"):
            raise ValueError(f"{body_id}: appearance/reference version mismatch")
        if not isinstance(appearance.get("description"), str) or len(appearance["description"]) < 30:
            raise ValueError(f"{body_id}: explicit appearance description required")
        anchors = appearance.get("anchors")
        if not isinstance(anchors, dict) or not anchors or any(
                not isinstance(v, str) or not v.strip() for v in anchors.values()):
            raise ValueError(f"{body_id}: named immutable appearance anchors required")
        if not isinstance(appearance.get("avoid", []), list) or any(
                not isinstance(v, str) for v in appearance.get("avoid", [])):
            raise ValueError("avoid must contain drawing instructions")
        references = []
        for index, ref in enumerate(source["references"]):
            path = resolve_asset(handoff / "assets", ref["file"])
            stats = inspect_image(path)
            if ref.get("sha256") != stats["sha256"]:
                raise ValueError(f"Reference changed: {ref['file']}")
            relative = f"references/{body_id}/{index:02}_{path.name}"
            native = ref.get("source_crop_pixels", [stats["width"], stats["height"]])
            if (not isinstance(native, list) or len(native) != 2 or any(
                    isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in native)):
                raise ValueError("Invalid native reference dimensions")
            item = {**copy.deepcopy(ref), "file": relative,
                    "stored_pixels": [stats["width"], stats["height"]],
                    "source_pixels": native,
                    "quality_warning": "Source crop was enlarged; no new detail was created."
                    if native != [stats["width"], stats["height"]] else None}
            references.append(item)
            copies.append((path, relative))
        staged_bodies[body_id] = {**copy.deepcopy(appearance), "references": references}
    # Validate every input before creating a partially populated project.
    output.mkdir(parents=True, exist_ok=True)
    for source, relative in copies:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    project = {"schema": 1, "kind": "studio_character_art_project",
               "source_registry_sha256": digest(registry_path),
               "source_spec_sha256": digest(spec_path), "style": spec["style"],
               "bodies": staged_bodies, "publication_approved": False}
    project["project_fingerprint"] = fingerprint(project)
    write_json(output / "project.json", project)
    return output / "project.json"


def load_project(root: Path) -> dict:
    project = read_json(root / "project.json")
    original = project.pop("project_fingerprint", None)
    if (project.get("kind") != "studio_character_art_project" or
            project.get("schema") != 1 or fingerprint(project) != original):
        raise ValueError("Character project changed; create a new version")
    project["project_fingerprint"] = original
    for body_id, body in project["bodies"].items():
        _id(body_id)
        for ref in body["references"]:
            path = resolve_asset(root, ref["file"])
            if digest(path) != ref["sha256"]:
                raise ValueError("Pinned character reference bytes changed")
    return project


def _references(body: dict, view: str) -> list[dict]:
    refs = body["references"]
    # Native pixels take precedence over enlarged contact-sheet crops. An exact
    # angle is useful only when it contains sufficient original detail.
    def priority(ref):
        native = min(ref["source_pixels"])
        return (native >= 512, ref["view"] == view, native)
    selected = sorted(refs, key=priority, reverse=True)[:3]
    return [{"file": r["file"], "sha256": r["sha256"], "view": r["view"],
             "role": "physical_body_identity", "source_pixels": r["source_pixels"],
             "review_status": r.get("review_status", "unreviewed"),
             "quality_warning": r.get("quality_warning")} for r in selected]


def _job(project: dict, body_ids: list[str], view: str, direction: str,
         *, job_id: str, episode: str | None = None) -> dict:
    references, paragraphs = [], []
    for body_id in body_ids:
        body = project["bodies"][body_id]
        paragraphs.append(f"PHYSICAL BODY {body_id}, version {body['identity_version']}: "
                          + body["description"] + "\nFIXED ANCHORS: "
                          + "; ".join(f"{k}: {v}" for k, v in body["anchors"].items())
                          + ".\nAVOID: " + "; ".join(body.get("avoid", [])))
        references.extend({**r, "body_id": body_id} for r in _references(body, view))
    prompt = ("Use case: identity-preserve. Production drawing for TWO WORLDS Studio.\n"
              "Style: " + project["style"] + "\n" + "\n".join(paragraphs) +
              "\nDRAW THIS: " + direction +
              "\nUse the supplied image references as the actual anatomy, face and wardrobe "
              "sources. Preserve physical left/right details; NEVER horizontally flip a "
              "character to obtain a new view. The same seed or prose alone does not lock "
              "identity. New poses must be drawn anatomically, not warped from a cutout. "
              "An exchanged consciousness changes acting, never the body's face or outfit. "
              "No typography, lettering, watermark, labels or extra characters. "
              "Maintain clear shapes, consistent material texture and believable anatomy. ")
    if references and view in REFERENCE_POSE:
        pose = REFERENCE_POSE[view]
        prompt = "EDIT THE REFERENCE INTO A NEW CAMERA VIEW. " + pose + "\n" + prompt
        prompt += "\nFINAL CAMERA REQUIREMENT: " + pose + " "
    if view in {"front", "three_quarter", "profile", "back", "rear_three_quarter"}:
        prompt += "Neutral studio background, whole figure inside canvas with 5% margin."
    sheet = view in {"expressions", "blink", "hands"}
    job = {"schema": 1, "id": _id(job_id), "body_ids": body_ids, "view": view,
           "project_fingerprint": project["project_fingerprint"],
           "prompt": prompt, "references": references,
           "output": {"target_pixels": [1536, 1024] if sheet else [1024, 1536],
                      "alpha_required": view == "rig_parts", "min_long_edge": 1024,
                      "single_pose": view in {"front", "three_quarter", "profile", "back", "rear_three_quarter"}},
           "review_status": "unreviewed", "publication_approved": False,
           "rig_ready": False, "identity_verified": False}
    if episode:
        job["episode"] = episode
    job["job_fingerprint"] = fingerprint(job)
    return job


def build_jobs(root: Path) -> dict:
    project = load_project(root)
    jobs = [_job(project, [body_id], view, instruction, job_id=f"{body_id}__{view}")
            for body_id in project["bodies"] for view, instruction in VIEWS.items()]
    result = {"schema": 1, "project_fingerprint": project["project_fingerprint"],
              "jobs": jobs, "images_generated": 0,
              "note": "Jobs require a drawing backend. A job is not generated artwork."}
    if (root / "jobs.json").exists():
        previous = read_json(root / "jobs.json")
        # Replanning is idempotent and preserves scene/revision jobs already
        # attached to this identity version, including their candidate lineage.
        if previous.get("project_fingerprint") != result["project_fingerprint"]:
            raise ValueError("Use a new plan for a new character identity version")
        planned_ids = {j["id"] for j in jobs}
        for job in previous.get("jobs", []):
            validate_job(root, job)
            if job["id"] not in planned_ids:
                result["jobs"].append(job)
    write_json(root / "jobs.json", result)
    return result


def save_job(root: Path, job: dict) -> dict:
    validate_job(root, job)
    plan = read_json(root / "jobs.json")
    existing = [j for j in plan["jobs"] if j["id"] == job["id"]]
    if existing and (len(existing) != 1 or existing[0] != job):
        raise ValueError("Job ID already belongs to another drawing contract")
    if not existing:
        plan["jobs"].append(job)
        write_json(root / "jobs.json", plan)
    return job


def scene_job(root: Path, episode: dict, scene_id: str, *, staging: dict | None = None) -> dict:
    project = load_project(root)
    matches = [s for s in episode.get("scenes", []) if s.get("id") == scene_id]
    if len(matches) != 1:
        raise ValueError("Scene must exist exactly once in the supplied Work screenplay")
    scene = matches[0]
    body_ids = scene.get("subjects", [])
    if not body_ids and scene.get("continuity", {}).get("body"):
        body_ids = [scene["continuity"]["body"]]
    if (not body_ids or not isinstance(body_ids, list) or len(set(body_ids)) != len(body_ids)
            or any(b not in project["bodies"] for b in body_ids)):
        raise ValueError("Every depicted physical body needs its own current reference set")
    direction = scene.get("shot")
    if not isinstance(direction, str) or len(direction) < 15:
        raise ValueError("Use a specific Work-authored shot direction")
    job = _job(project, body_ids, "scene", direction,
               job_id=f"{_id(episode['id'])}__{_id(scene_id)}", episode=episode["id"])
    job.pop("job_fingerprint")
    job["source_scene_sha256"] = fingerprint(scene)
    job["source_episode_sha256"] = fingerprint(episode)
    if staging is not None:
        if not isinstance(staging, dict) or any(b not in body_ids for b in staging):
            raise ValueError("Staging may only describe physical bodies in the supplied scene")
        directions = []
        for body_id, placement in staging.items():
            if not isinstance(placement, dict) or set(placement)-{"view", "facing", "plane"}:
                raise ValueError("Staging supports view, facing and plane only")
            if placement.get("view") not in {"front", "three_quarter", "profile", "back", "rear_three_quarter"}:
                raise ValueError("Use an explicit drawn camera angle")
            if placement.get("facing") not in {"mirror", "camera", "left", "right", "away"}:
                raise ValueError("Use an explicit facing direction")
            if placement.get("plane") not in {"foreground", "beyond_glass", "midground"}:
                raise ValueError("Use an explicit scene depth plane")
            directions.append(f"{body_id}: {placement['view']} view; faces {placement['facing']}; "
                              f"depth plane {placement['plane']}")
        job["staging"] = copy.deepcopy(staging)
        job["prompt"] += "\nEXPLICIT CAMERA STAGING: " + "; ".join(directions) + "."
    job["job_fingerprint"] = fingerprint(job)
    return job


def validate_job(root: Path, job: dict) -> None:
    project = load_project(root)
    checked = copy.deepcopy(job)
    expected = checked.pop("job_fingerprint", None)
    if fingerprint(checked) != expected or job.get("project_fingerprint") != project["project_fingerprint"]:
        raise ValueError("Drawing job or source identity version changed")
    for ref in job["references"]:
        if digest(resolve_asset(root, ref["file"])) != ref["sha256"]:
            raise ValueError("Drawing reference changed")
        known = project["bodies"].get(ref["body_id"], {}).get("references", [])
        if not any(r["file"] == ref["file"] and r["sha256"] == ref["sha256"] for r in known):
            raise ValueError("Job contains an unrelated character reference")
    if job.get("edit_target"):
        target = job["edit_target"]
        if digest(resolve_asset(root, target["file"])) != target["sha256"]:
            raise ValueError("Revision target changed")


def get_job(root: Path, job_id: str) -> dict:
    _id(job_id)
    plan = read_json(root / "jobs.json")
    matches = [j for j in plan.get("jobs", []) if j.get("id") == job_id]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicated drawing job")
    validate_job(root, matches[0])
    return matches[0]


def register_candidate(root: Path, job_id: str, source: Path, *, generator: str,
                       note: str) -> dict:
    job = get_job(root, job_id)
    if not generator.strip() or not note.strip():
        raise ValueError("Record the real generator and execution provenance")
    stats = inspect_image(source)
    with Image.open(source) as im:
        im.load()
        if im.format != "PNG":
            raise ValueError("Save the original generated PNG; do not relabel a different format")
        alpha = im.convert("RGBA").getchannel("A")
        occupied = alpha.getbbox()
        real_transparency = bool(alpha.getextrema()[0] < 255 and occupied)
    problems = []
    if max(stats["width"], stats["height"]) < job["output"]["min_long_edge"]:
        problems.append("Native generated resolution is below the drawing target; upscaling cannot fix it.")
    if job["output"]["alpha_required"] and not real_transparency:
        problems.append("True transparent background is required; an RGBA label alone is insufficient.")
    if stats["alpha"] and not occupied:
        problems.append("Image is completely transparent.")
    if stats["contrast"] < .015:
        problems.append("Almost uniform image: inspect missing artwork or failed generation.")
    if job["output"]["single_pose"] and stats["width"] / stats["height"] > 1.1:
        problems.append("Wide image for a single full-body pose; inspect framing/cropped feet.")
    candidate_id = stats["sha256"][:16]
    relative = f"candidates/{job_id}/{candidate_id}.png"
    target = root / relative
    metadata_path = target.with_suffix(".json")
    if metadata_path.exists():
        existing = read_json(metadata_path)
        if digest(target) != stats["sha256"] or existing["job_fingerprint"] != job["job_fingerprint"]:
            raise ValueError("Existing candidate/source contract changed")
        return existing
    if target.exists():
        raise FileExistsError("Candidate bytes exist without provenance; inspect before importing")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    result = {"schema": 1, "id": candidate_id, "job_id": job_id,
              "job_fingerprint": job["job_fingerprint"], "image": relative,
              "body_ids": job["body_ids"], "view": job["view"],
              "generator": generator, "provenance_note": note,
              "technical": {**stats, "real_transparency": real_transparency,
                            "issues": problems, "pass": not problems},
              "human_review": None, "review_status": "unreviewed",
              "identity_verified": False, "rig_ready": False,
              "publication_approved": False}
    write_json(metadata_path, result)
    return result


def candidates(root: Path) -> list[tuple[Path, dict]]:
    load_project(root)
    result = []
    for path in sorted((root / "candidates").glob("*/*.json")):
        item = read_json(path)
        job = get_job(root, item["job_id"])
        if item["job_fingerprint"] != job["job_fingerprint"]:
            raise ValueError("Candidate was produced against another drawing contract")
        if digest(resolve_asset(root, item["image"])) != item["technical"]["sha256"]:
            raise ValueError("Candidate artwork changed since review")
        result.append((path, item))
    return result


def _validate_review(item: dict, review: dict) -> None:
    if review.get("decision") not in {"accept", "revise"}:
        raise ValueError("Human review decision must be accept or revise")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("Name the actual human reviewer")
    checks = review.get("criteria", {})
    if set(checks) != set(CRITERIA) or any(type(v) is not bool for v in checks.values()):
        raise ValueError("Explicitly assess all seven visual criteria as true/false")
    if not isinstance(review.get("notes"), str) or not review["notes"].strip():
        raise ValueError("Record concrete visual review notes")
    if review.get("image_sha256") != item["technical"]["sha256"]:
        raise ValueError("Review must identify the actual candidate image SHA-256")
    if review["decision"] == "accept" and (not all(checks.values()) or not item["technical"]["pass"]):
        raise ValueError("Resolve technical and visual findings before accepting this drawing")


def record_review(root: Path, job_id: str, candidate_id: str, review: dict) -> dict:
    matches = [(p, c) for p, c in candidates(root)
               if c["job_id"] == job_id and c["id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("Candidate not found exactly once")
    path, item = matches[0]
    _validate_review(item, review)
    # This command records a supplied human decision. It is never invoked as
    # an automatic inference step and never grants publication/rig approval.
    item["human_review"] = copy.deepcopy(review)
    item["review_status"] = "accepted_drawing" if review["decision"] == "accept" else "needs_revision"
    write_json(path, item)
    return item


def request_revision(root: Path, job_id: str, candidate_id: str, *,
                     reported_by: str, notes: str) -> dict:
    """Record concrete defects without impersonating human approval."""
    if not reported_by.strip() or not notes.strip():
        raise ValueError("Identify who observed the concrete drawing defect")
    matches = [(p,c) for p,c in candidates(root)
               if c["job_id"] == job_id and c["id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("Candidate not found exactly once")
    path, item = matches[0]
    item["revision_request"] = {"reported_by": reported_by, "notes": notes,
                                "image_sha256": item["technical"]["sha256"]}
    item["review_status"] = "needs_revision"
    write_json(path, item)
    return item


def import_reviews(root: Path, bundle: dict) -> dict:
    if bundle.get("kind") != "human_review_drafts" or not isinstance(bundle.get("reviews"), list):
        raise ValueError("Use the explicit human review export from the drawing gallery")
    known = {(c["job_id"], c["id"]): (p, c) for p, c in candidates(root)}
    staged, seen = [], set()
    for review in bundle["reviews"]:
        key = (review.get("job_id"), review.get("candidate_id"))
        if key not in known or key in seen:
            raise ValueError("Unknown or duplicate reviewed candidate")
        seen.add(key)
        path, item = known[key]
        _validate_review(item, review)
        item["human_review"] = copy.deepcopy(review)
        item["review_status"] = "accepted_drawing" if review["decision"] == "accept" else "needs_revision"
        staged.append((path, item))
    # No review is written until the whole supplied batch passes validation.
    for path, item in staged:
        write_json(path, item)
    return {"reviews_recorded": len(staged), "publication_approved": False}


def project_status(root: Path) -> dict:
    project = load_project(root)
    items = [c for _, c in candidates(root)]
    coverage = {}
    for body_id in project["bodies"]:
        present = {c["view"] for c in items if c["body_ids"] == [body_id]}
        coverage[body_id] = {"drawn_views": sorted(present),
                             "missing_views": sorted(set(VIEWS) - present)}
    return {"schema": 1, "candidate_drawings": len(items),
            "technical_pass": sum(c["technical"]["pass"] for c in items),
            "human_accepted_drawings": sum(c["review_status"] == "accepted_drawing" for c in items),
            "coverage": coverage, "identity_automatically_verified": False,
            "rig_ready": False, "publication_approved": False}


def revision_job(root: Path, job_id: str, candidate_id: str) -> dict:
    item = next((c for _, c in candidates(root)
                 if c["job_id"] == job_id and c["id"] == candidate_id), None)
    if item is None or item.get("review_status") != "needs_revision":
        raise ValueError("A concrete revision review is required")
    job = copy.deepcopy(get_job(root, job_id))
    job.pop("job_fingerprint")
    job["id"] = _id(job_id[:68] + "__r" + candidate_id[:12])
    job["revision_of"] = item["technical"]["sha256"]
    job["edit_target"] = {"file": item["image"], "sha256": item["technical"]["sha256"]}
    feedback = item.get("revision_request") or item.get("human_review")
    if not feedback:
        raise ValueError("Concrete visual findings are required for a targeted revision")
    job["prompt"] += "\nCORRECT THIS SPECIFIC CANDIDATE: " + feedback["notes"]
    job["job_fingerprint"] = fingerprint(job)
    return save_job(root, job)


def _preview(path: Path, edge: int = 1100) -> str:
    with Image.open(path) as im:
        im.load()
        im = im.convert("RGBA")
        im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def export_gallery(root: Path, output: Path) -> Path:
    """Self-contained private HTML report; no external scripts/fonts/art URLs."""
    project = load_project(root)
    cards = []
    for _, item in candidates(root):
        job = get_job(root, item["job_id"])
        images = [{**r, "src": _preview(resolve_asset(root, r["file"]), 700)}
                  for r in job["references"]]
        cards.append({**item, "src": _preview(resolve_asset(root, item["image"]), 4096),
                      "references": images})
    payload = json.dumps({"cards": cards, "criteria": CRITERIA,
                          "bodies": list(project["bodies"])}, ensure_ascii=False).replace("<", "\\u003c")
    template = (Path(__file__).resolve().parent.parent / "templates/character_art_review.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__CHARACTER_DATA__", payload), encoding="utf-8")
    return output


def reference_conditioned(workflow: dict) -> bool:
    """Check reference image reaches a sampler input, not an unused LoadImage.

    Graph reachability is a structural guard, not a semantic identity guarantee.
    The running provider still needs its own installed reference-aware model.
    """
    def refs(value):
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
            return [value[0]] if value[0] in workflow else []
        if isinstance(value, dict):
            return [n for v in value.values() for n in refs(v)]
        if isinstance(value, list):
            return [n for v in value for n in refs(v)]
        return []
    def ancestors(start):
        todo, seen = list(start), set()
        while todo:
            node = todo.pop()
            if node not in seen:
                seen.add(node)
                todo.extend(refs(workflow[node].get("inputs", {})))
        return seen
    active = ancestors(["9"]) if "9" in workflow else set()
    for node_id in active:
        node = workflow[node_id]
        if "sampler" not in str(node.get("class_type", "")).lower():
            continue
        upstream = ancestors(refs(node.get("inputs", {})))
        if any(workflow[n].get("class_type") == "LoadImage" and
               workflow[n].get("inputs", {}).get("image") == "__REFERENCE_IMAGE__"
               for n in upstream):
            return True
    return False


def render_comfyui(root: Path, job_id: str, *, workflow_path: Path,
                   checkpoint: str, endpoint: str) -> dict:
    from studio.art_director import upload_reference, comfy_workflow, comfy_render
    job = get_job(root, job_id)
    if job.get("edit_target"):
        raise ValueError("Targeted revision needs a separate edit input; use the exported edit job")
    if len(job["body_ids"]) != 1:
        raise ValueError("Multi-body shots need an explicitly isolated multi-reference backend")
    template = read_json(workflow_path)
    if not reference_conditioned(template):
        raise ValueError("Workflow must connect the identity reference to the active sampler")
    primary = job["references"][0]
    # Temporary output keeps private artwork outside the public repository and
    # registration preserves all versions rather than overwriting a candidate.
    with tempfile.TemporaryDirectory(prefix="tw-drawing-") as tmp:
        name = upload_reference(endpoint, resolve_asset(root, primary["file"]))
        prompt = {"prompt": job["prompt"], "negative": "watermark, duplicate anatomy, blurred face",
                  "seed": int(job["job_fingerprint"][:8], 16),
                  "episode": "CHARACTER_ART", "scene": job_id}
        width, height = job["output"]["target_pixels"]
        workflow = comfy_workflow(prompt, template, checkpoint, width, height, name)
        image = Path(tmp) / "drawing.png"
        result = comfy_render(workflow, endpoint, image)
        return register_candidate(root, job_id, image, generator="comfyui_reference_conditioned",
                                  note="Executed ComfyUI job " + result["comfy_prompt_id"])


def main():
    parser = argparse.ArgumentParser(description="Studio character drawing desk: private references, drawing and visual review")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--handoff", type=Path, required=True)
    init.add_argument("--spec", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    for command in ("plan", "status", "register", "gallery", "review", "review-batch", "request-revision", "revision", "scene", "render-comfyui"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--project", type=Path, required=True)
        if command in {"register", "review", "request-revision", "revision", "render-comfyui"}:
            cmd.add_argument("--job", required=True)
        if command in {"review", "request-revision", "revision"}:
            cmd.add_argument("--candidate", required=True)
        if command in {"gallery", "scene", "revision"}:
            cmd.add_argument("--output", type=Path, required=True)
        if command == "register":
            cmd.add_argument("--image", type=Path, required=True)
            cmd.add_argument("--generator", required=True)
            cmd.add_argument("--note", required=True)
        if command in {"review", "review-batch"}:
            cmd.add_argument("--review", type=Path, required=True)
        if command == "request-revision":
            cmd.add_argument("--reported-by", required=True)
            cmd.add_argument("--notes", required=True)
        if command == "scene":
            cmd.add_argument("--episode", type=Path, required=True)
            cmd.add_argument("--scene", required=True)
            cmd.add_argument("--staging", type=Path, help="Explicit camera angle/facing/depth per depicted physical body")
        if command == "render-comfyui":
            cmd.add_argument("--workflow", type=Path, required=True)
            cmd.add_argument("--checkpoint", required=True)
            cmd.add_argument("--url", required=True)
    a = parser.parse_args()
    if a.command == "init":
        result = {"project": str(create_project(a.handoff, a.spec, a.output))}
    elif a.command == "plan":
        plan = build_jobs(a.project)
        result = {"jobs": len(plan["jobs"]), "images_generated": 0, "plan": str(a.project / "jobs.json")}
    elif a.command == "status":
        result = project_status(a.project)
    elif a.command == "register":
        result = register_candidate(a.project, a.job, a.image, generator=a.generator, note=a.note)
    elif a.command == "gallery":
        result = {"gallery": str(export_gallery(a.project, a.output))}
    elif a.command == "review":
        result = record_review(a.project, a.job, a.candidate, read_json(a.review))
    elif a.command == "review-batch":
        result = import_reviews(a.project, read_json(a.review))
    elif a.command == "request-revision":
        result = request_revision(a.project,a.job,a.candidate,reported_by=a.reported_by,notes=a.notes)
    elif a.command == "revision":
        result = revision_job(a.project, a.job, a.candidate)
        write_json(a.output, result)
    elif a.command == "scene":
        result = save_job(a.project, scene_job(a.project, read_json(a.episode), a.scene,
                          staging=read_json(a.staging) if a.staging else None))
        write_json(a.output, result)
    else:
        result = render_comfyui(a.project, a.job, workflow_path=a.workflow,
                                checkpoint=a.checkpoint, endpoint=a.url)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
