"""Elizabeth Bones EP01 S00-S06. Private, real TWO WORLDS Studio preview.

This module is the orchestrator and provenance gate; the FINAL MP4 is rendered
by studio.hybrid_master, never by InVideo, Runway, or a remote video editor.
All GPU media remains inside one private Kaggle kernel. No auto publishing.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import subprocess
import zipfile
from pathlib import Path

from studio.kaggle_api import ROOT
from studio.kaggle_identity import HANDLE
from studio.release_gate import sha256

SCENES = (
    # S00 is a title shot, the next six cues are from the locked Russian script.
    ("S00", "", "", "establishing", "push_in", 2.2),
    ("S01", "ЭЛИЗАБЕТ", "Следующий. Имя?", "over_shoulder", "pan_right", 2.1),
    ("S02", "КОСТЯ", "Константин Мортисович Костин. Для друзей — Костя.", "medium", "push_in", 4.25),
    ("S03", "ЭЛИЗАБЕТ", "Возраст?", "close", "pull_out", 1.6),
    ("S04", "КОСТЯ", "Двести три. Но душой я мальчишка!", "medium", "pan_left", 3.05),
    ("S05", "ЭЛИЗАБЕТ", "Вижу. А где душа?", "close", "push_in", 2.35),
    ("S06", "КОСТЯ", "На испытательном. Уже три века.", "medium", "pan_right", 2.95),
)
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PACK = "ELIZABETH_BONES_STUDIO_PILOT.zip"
RECEIPT = "ELIZABETH_BONES_STUDIO_JOB.json"
WORKER = "tools/kaggle_elizabeth_pilot_entry.py"
WIDTH, HEIGHT, FPS = 576, 1024, 24

# Character descriptions always repeated to reduce face/wardrobe drift. This
# does not certify identity: user review is mandatory before any release.
SETTING = (
    "High-end cinematic realistic still photography, warm brass practical lamp, "
    "deep burgundy velvet curtains, expensive walnut wood desk, one red rose, "
    "dark gothic funeral office without bodies, humane respectful atmosphere, "
    "soft key light, realistic human skin, subtle 35mm film grain, "
    "rich tonal range, naturally detailed black fabrics. "
)
LIZ = (
    "Elizabeth Bones, one consistent adult dark-haired woman with long dark "
    "wavy hair, precise eyebrows, natural expressive brown eyes, pale natural "
    "skin texture, elegant fitted black blouse with delicate black lace sleeves. "
)
KOSTYA = (
    "Kostya, one coherent adult articulated anatomically correct human skeleton "
    "with matte ivory bone, black tailored three-piece suit, clean light shirt, "
    "burgundy tie, black fedora, anatomically coherent hands and jaw. "
)
NEGATIVE = (
    "text, letters, logos, watermark, plastic skin, low quality, blurry, deformed, "
    "duplicate body, extra hands, extra fingers, detached body parts, merged bones, "
    "twisted legs, skull morphing, broken jaw, missing jacket, open shirt, "
    "exaggerated cartoon proportions, corpse, blood, nudity"
)
SHOT_PROMPTS = (
    SETTING + LIZ + KOSTYA +
    "Wide vertical shot: Elizabeth seated LEFT behind desk, Kostya standing RIGHT "
    "by a leather chair, full room geography, wide dramatic cinematic establishing frame.",
    SETTING + LIZ +
    "Medium close over-the-shoulder interview shot looking at Elizabeth; she sits "
    "upright behind closed folder, one hand holding a pen, deadpan curious expression, "
    "foreground blurred brim of a black fedora, interview axis LEFT to RIGHT.",
    SETTING + KOSTYA +
    "Medium chest-up opposite interview shot of Kostya standing by chair on RIGHT, "
    "complete immaculate black vest and jacket, holds paper resume, slight confident "
    "posture, stays entirely human anatomically correct skeletal form.",
    SETTING + LIZ +
    "Strong cinematic close-up side angle of Elizabeth's hand holding pen frozen "
    "above cream paper, her thoughtful profile in background, elegant lace visible.",
    SETTING + KOSTYA +
    "Side medium shot Kostya already seated confidently, RIGHT of table, "
    "both shoulders naturally placed under tailored coat, one open bony palm, "
    "dryly amused body language, clean anatomical hand.",
    SETTING + LIZ +
    "Fresh independent three-quarter close portrait of Elizabeth, left cheek "
    "resting lightly on one hand, raised eyebrow, dry amused expression, "
    "matte black blouse lace visible, natural lips closed.",
    SETTING + KOSTYA +
    "Distinct over-table medium profile shot of Kostya, he adjusts the brim "
    "of black fedora with one bony hand, complete suit, dry confident pose "
    "near brass lamp, visible burgundy tie, coherent jaw."
)


def _duration(seconds: float, minimum: float) -> float:
    if not math.isfinite(seconds) or not .15 < seconds < 28:
        raise ValueError("Natural voice file is silent, corrupt or unexpectedly long")
    return math.ceil(max(minimum, seconds + .24) * FPS) / FPS


def build_manifest(work_dir: Path, voice_durations: dict[str, float]) -> dict:
    """Build the precise TWO WORLDS native hybrid_master shot manifest."""
    shots = []
    for scene_id, speaker, text, shot_type, camera, minimum in SCENES:
        art = work_dir / "art" / (scene_id + ".png")
        if not art.is_file() or art.stat().st_size < 5000:
            raise FileNotFoundError("Real GPU art missing: " + scene_id)
        shot = {
            "image": "art/" + scene_id + ".png", "shot_type": shot_type,
            "camera": camera, "camera_strength": .34,
            "caption_position": "bottom" if speaker else "top",
        }
        if speaker:
            voice_file = work_dir / "voices" / (scene_id + ".wav")
            if not voice_file.is_file() or voice_file.stat().st_size < 2000:
                raise FileNotFoundError("Real Qwen voice missing: " + scene_id)
            seconds = voice_durations.get(scene_id)
            if type(seconds) not in (int, float):
                raise ValueError("Missing real speech clock: " + scene_id)
            shot.update(
                duration=_duration(float(seconds), minimum),
                voice="voices/" + scene_id + ".wav",
                caption=text, dialogue_text=text,
                voice_bus="dialogue",
                approved=False, voice_approved=False,
            )
        else:
            shot.update(duration=minimum, caption="", approved=False)
        shots.append(shot)
    score = work_dir / "score.wav"
    if not score.is_file() or score.stat().st_size < 3000:
        raise FileNotFoundError("Private authored soundtrack missing")
    return {
        "schema": 1, "width": WIDTH, "height": HEIGHT, "fps": FPS,
        "language": "ru", "burn_subtitles": True,
        "music": "score.wav", "music_volume": .10, "shots": shots,
        "editorial_approved": False, "rights_approved": False,
        "review_policy": None,  # dropped before calling hybrid_master
    }


def native_render(work_dir: Path, voice_durations: dict[str, float]) -> dict:
    """No external editor; actual render is studio.hybrid_master.produce."""
    from studio.hybrid_master import produce
    manifest = build_manifest(work_dir, voice_durations)
    manifest.pop("review_policy")
    file = work_dir / "studio_shot_manifest.json"
    file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = work_dir / "ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.mp4"
    receipt = produce(file, output, master=False, quality_check=False)
    if receipt["shots"] != len(SCENES) or receipt["static_camera_shots"] != len(SCENES):
        raise RuntimeError("Native Studio did not actually render all seven shots")
    if receipt["audio_tracks"] != 7:  # six real voice tracks + one generated score
        raise RuntimeError("Native Studio missed required voice/music tracks")
    if not receipt["captions_burned_in"] or receipt["published"]:
        raise RuntimeError("Wrong Studio review-only render profile")
    return receipt


def bundle(root: Path = ROOT) -> bytes:
    root = root.resolve()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((root / "studio").glob("*.py")):
            real = path.resolve()
            if not path.is_file() or path.is_symlink() or real.parent != (root / "studio"):
                raise ValueError("Unsafe Studio source path")
            if path.stat().st_size > 180_000:
                raise ValueError("Source file too large for worker")
            z.write(path, path.relative_to(root).as_posix())
    if not 10_000 < buf.tell() < 2_000_000:
        raise ValueError("Native Studio source archive unexpected size")
    return buf.getvalue()


def prepare(username: str, nonce: str, destination: Path, *,
            root: Path = ROOT, revision: str | None = None) -> dict:
    if not HANDLE.fullmatch(username):
        raise ValueError("Invalid Kaggle username")
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,90}", nonce):
        raise ValueError("Invalid private run nonce")
    rev = revision or os.environ.get("GITHUB_SHA", "local-contract")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", rev):
        raise ValueError("Invalid reviewed source revision")
    template = (Path(root) / WORKER).read_text(encoding="utf-8")
    if template.count("__ELIZABETH_CONFIG_B64__") != 1 or template.count("__ELIZABETH_SOURCE_B64__") != 1:
        raise ValueError("Wrong native Studio worker template")
    source = bundle(Path(root))
    payload = {
        "schema": 1, "run_nonce": nonce, "revision": rev,
        "fps": FPS, "width": WIDTH, "height": HEIGHT,
        "image_model": IMAGE_MODEL, "voice_model": VOICE_MODEL,
        "scenes": [r[0] for r in SCENES],
        "lock_publishing": True
    }
    slug = "tw-elizabeth-native-" + hashlib.sha256(nonce.encode()).hexdigest()[:12]
    metadata = {
        "id": username + "/" + slug, "title": slug,
        "code_file": "kernel_main.py", "language": "python",
        "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    code = template.replace(
        "__ELIZABETH_CONFIG_B64__",
        base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode("ascii")
    ).replace(
        "__ELIZABETH_SOURCE_B64__", base64.b64encode(source).decode("ascii")
    )
    (destination / "kernel_main.py").write_text(code, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return {"kernel": metadata["id"], "nonce": nonce,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "shots": len(SCENES), "real_gpu_execution": False,
            "private_review_only": True}


def verify_outputs(output_folder: Path, nonce: str) -> dict:
    receipts = list(output_folder.rglob(RECEIPT))
    packages = list(output_folder.rglob(PACK))
    if len(receipts) != 1 or len(packages) != 1:
        raise FileNotFoundError("No unique private Studio result/manifest")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    archive = packages[0]
    if (receipt.get("schema") != 1 or receipt.get("run_nonce") != nonce
            or receipt.get("success") is not True
            or receipt.get("pipeline") != "studio.hybrid_master.produce"
            or receipt.get("provider") != "native_two_worlds_studio"
            or receipt.get("shots") != 7 or receipt.get("voices") != 6
            or receipt.get("published") is not False
            or receipt.get("editorial_approved") is not False):
        raise ValueError("No verified 7-shot native Studio production")
    if not (20_000 < archive.stat().st_size < 300_000_000):
        raise ValueError("Incorrect private preview archive size")
    if receipt.get("archive_sha256") != sha256(archive):
        raise ValueError("Private native Studio archive mismatch")
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        expected = {
            "ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.mp4",
            "ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.srt",
            "ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.json",
            "studio_shot_manifest.json",
            "production_provenance.json"
        }
        if names != expected or any(
            i.file_size > 250_000_000 or i.is_dir()
            or (i.external_attr >> 16) & 0o170000 == 0o120000
            for i in z.infolist()
        ):
            raise ValueError("Unexpected Studio export files")
        video = z.read("ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.mp4")
        rendered = json.loads(z.read("ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.json"))
        proof = json.loads(z.read("production_provenance.json"))
        srt = z.read("ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.srt")
    digest = hashlib.sha256(video).hexdigest()
    if (digest != receipt.get("video_sha256")
            or digest != rendered.get("video_sha256")
            or proof.get("studio_renderer") != "studio.hybrid_master"
            or proof.get("image_model") != IMAGE_MODEL
            or proof.get("voice_model") != VOICE_MODEL
            or not srt or b"\xd0" not in srt):
        raise ValueError("Export hashes, provenance or Russian subtitles do not match")
    return {"status": "NATIVE_STUDIO_GPU_RENDER_VERIFIED_REVIEW_REQUIRED",
            "video_sha256": digest, "seconds": receipt.get("duration"),
            "shots": 7, "voice_cues": 6, "external_editor": False,
            "human_approved": False, "published": False}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    a = s.add_parser("prepare")
    a.add_argument("--username", required=True)
    a.add_argument("--nonce", required=True)
    a.add_argument("--destination", type=Path, required=True)
    b = s.add_parser("verify")
    b.add_argument("--path", type=Path, required=True)
    b.add_argument("--nonce", required=True)
    args = p.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.username, args.nonce, args.destination)
    else:
        result = verify_outputs(args.path, args.nonce)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
