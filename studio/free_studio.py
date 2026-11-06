"""Free-first production dispatcher: deterministic planning, NEVER hidden inference.

No scripts are authored or changed. Does not schedule public posts. Presence of a
token means configured, not tested, and never automatically authorizes spending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from studio.cloudflare_flux import MODEL as FLUX_MODEL
from studio.cloudflare_flux import plan as flux_plan

SECRET_KEYS = {
    "kaggle": ("KAGGLE_API_TOKEN",),
    "hugging_face": ("HF_TOKEN",),
    "cloudflare_image": ("CLOUDFLARE_AI_TOKEN",),
    "gemini": ("GEMINI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "horde": ("HORDE_API_KEY",),
    "groq_optional": ("GROQ_API_KEY",),
    "azure_optional": ("AZURE_SPEECH_KEY",),
    "modal_optional": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
    "r2_optional": ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"),
}
VARIABLE_KEYS = {
    "cloudflare_image": ("CLOUDFLARE_ACCOUNT_ID",),
    "azure_optional": ("AZURE_SPEECH_REGION",),
    "cloudflare_d1_optional": ("CLOUDFLARE_D1_DATABASE_ID",),
    "cloudflare_queue_optional": ("CLOUDFLARE_QUEUE_NAME",),
    "r2_optional": ("R2_BUCKET_NAME",),
}
LOCAL_MODULES = {
    "montage": "studio.hybrid_master",
    "camera": "studio.smooth_camera",
    "continuity_metadata": "studio.continuity_ledger",
    "batch_release_gate": "studio.triptych_batch",
    "comfyui_kaggle": "studio.comfy_kaggle",
    "narration_wav": "studio.narration_batch",
    "reference_graph": "studio.comfy_reference_guard",
}
FUTURE_ADAPTERS = ("ace_step_1_5", "depth_anything_v2_small", "sam2", "wan2_1",
                   "piper_ru", "blender_2_5d", "whisper_cpp")
SCENE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,64}$")


def readiness(env: dict[str, str] | None = None) -> dict:
    env = os.environ if env is None else env
    status = {}
    for key, keys in SECRET_KEYS.items():
        has_secret = all(isinstance(env.get(name), str) and bool(env.get(name).strip())
                         for name in keys)
        needs = VARIABLE_KEYS.get(key, ())
        has_vars = all(bool(env.get(name, "").strip()) for name in needs)
        configured = has_secret and has_vars
        status[key] = {
            "configured": configured,
            "credentials_present": has_secret,
            "required_settings_present": has_vars,
            "authenticated": None,   # Must be tested against real API separately.
            "inference_verified": False,
        }
    for name, keys in VARIABLE_KEYS.items():
        if name not in status:
            status[name] = {"configured": all(bool(env.get(x, "").strip()) for x in keys),
                            "authenticated": None, "inference_verified": False}
    return {"schema": 1, "services": status,
            "already_in_source": LOCAL_MODULES,
            "models_not_installed_or_tested": list(FUTURE_ADAPTERS),
            "execution_default": "dry_run",
            "automated_publication": False,
            "paid_fallback": False}


def scene_route(scene: dict, status: dict, *, public_generic: bool = False,
                width: int = 1024, height: int = 1792) -> dict:
    """Choose a safe image lane without silently transmitting owner IP.

    For confidential cinematic content use only user-approved cloud providers.
    Horde is voluntary third-party hardware, thus only for explicit public
    generic jobs with no identity references. Never assume Kaggle readback works.
    """
    proposal = flux_plan(scene, width=width, height=height)
    references = scene.get("references") or {}
    if not isinstance(references, dict):
        raise ValueError("Scene references must be an object")
    private = not public_generic
    if status.get("cloudflare_image", {}).get("configured"):
        provider, reason = "cloudflare_image", "Configured; inference is opt-in and unverified"
    elif public_generic and not references and status.get("horde", {}).get("configured"):
        provider, reason = "horde", "Generic public artwork only; worker privacy not guaranteed"
    else:
        provider, reason = None, "No safe configured image generator; preserve job for later"
    return {"scene": scene["scene"], "provider": provider,
            "reason": reason, "private": private,
            "reference_count": len(references),
            "requires_owner_reference_approval": bool(references),
            "prompt_sha256": proposal["prompt_sha256"],
            "generator": FLUX_MODEL if provider == "cloudflare_image" else None,
            "automatic_submission": False, "published": False, "art_review_required": True,
            "note": "Kaggle is managed separately; its prior HTTP429 readback requires proof."}


def prepare_batch(prompt_pack: dict, env: dict[str, str] | None = None,
                  *, public_generic: bool = False) -> dict:
    if prompt_pack.get("schema") != 1 or not isinstance(prompt_pack.get("scenes"), list):
        raise ValueError("Expected compiled schema=1 prompt pack; never rewrite a screenplay")
    scenes = prompt_pack["scenes"]
    if not scenes or len(scenes) > 500:
        raise ValueError("Expected 1..500 already-authored scene prompts")
    ids = [scene.get("scene") for scene in scenes if isinstance(scene, dict)]
    if len(ids) != len(scenes) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate or invalid scene jobs")
    for sid in ids:
        if not isinstance(sid, str) or not SCENE_ID.fullmatch(sid):
            raise ValueError("Unsafe scene job identifier")
    status = readiness(env)["services"]
    rows = [scene_route(s, status, public_generic=public_generic) for s in scenes]
    public_statement = ("Only generic unowned public reference-free scenes."
                        if public_generic else "Unpublished canonical story assets remain confidential.")
    payload = json.dumps({"episode": prompt_pack.get("episode"),
                          "scenes": [{"scene":s.get("scene"), "prompt":s.get("prompt"),
                                      "seed":s.get("seed")} for s in scenes]},
                         sort_keys=True, ensure_ascii=False)
    return {"schema": 1, "episode": prompt_pack.get("episode"),
            "source_prompt_pack_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "scene_count": len(rows), "routes": rows,
            "safe_free_only": True, "requested_cost_ceiling_usd": 0,
            "external_account_zero_charge_guaranteed": False,
            "note": "Zero-dollar policy is enforced by not automatically invoking providers; "
                    "external plan/quota changes must be checked before execution.",
            "privacy_policy": public_statement,
            "script_modified": False, "auto_publish": False}


def main(argv=None):
    p = argparse.ArgumentParser(description="Offline configured-provider audit or dry-run visual routes")
    p.add_argument("--prompt-pack", type=Path)
    p.add_argument("--public-generic", action="store_true",
                   help="Only for genuinely public generic scenes without private references")
    args = p.parse_args(argv)
    if args.prompt_pack:
        pack = json.loads(args.prompt_pack.read_text(encoding="utf-8"))
        result = prepare_batch(pack, public_generic=args.public_generic)
    else:
        result = readiness()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
