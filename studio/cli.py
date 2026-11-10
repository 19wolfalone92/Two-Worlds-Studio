"""TWO WORLDS single CLI for safe preflight, voice routing and review.

Legacy entrypoints continue working. External compute/paid quota access is
explicit; default commands are offline/read-only and never publish anything.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ENGINES = {
    "comfy": "studio.comfy_kaggle",
    "kaggle": "studio.kaggle_api",
    "art": "studio.art_director",
    "kokoro": "studio.kokoro_tts",
    "qwen": "studio.russian_qwen_tts",
    "batch": "studio.triptych_batch",
    "master": "studio.hybrid_master",
    "lipsync": "studio.lipsync",
    "blender": "studio.blender_clip",
    "character": "studio.neural_character",
}
PROVIDER_ENGINES = {"comfy", "kaggle", "art", "kokoro", "qwen"}


def doctor() -> dict:
    return {
        "status": "offline_inventory", "python": sys.version.split()[0],
        "binaries": {k: shutil.which(k) is not None
                     for k in ("ffmpeg", "ffprobe", "kaggle", "nvidia-smi", "rhubarb", "blender")},
        "optional_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("pydantic", "soundfile", "torch", "kokoro", "cryptography")
        },
        "connections_configured_not_verified": {
            name: bool(os.environ.get(name))
            for name in ("KAGGLE_API_TOKEN", "HF_TOKEN", "CLOUDFLARE_AI_TOKEN")
        },
        "network_used": False, "gpu_used": False,
        "models_loaded": False, "published": False
    }


def run_engine(engine: str, arguments: list[str], *,
               execute: bool = False, allow_provider_usage: bool = False) -> dict:
    if engine not in ENGINES:
        raise ValueError("Unknown engine")
    # '--' separates wrapper options from driver options.
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if len(arguments) > 60 or any(not isinstance(arg, str) or len(arg) > 2000 for arg in arguments):
        raise ValueError("Unbounded driver arguments")
    if not execute:
        return {"status": "planned", "engine": engine, "module": ENGINES[engine],
                "argument_count": len(arguments), "actual_execution": False,
                "provider_usage_allowed": False, "published": False}
    if engine in PROVIDER_ENGINES and not allow_provider_usage:
        raise ValueError("Provider/model execution requires --allow-provider-usage")
    # Process vector, NEVER a shell string. Driver itself performs its own
    # license, privacy, no-overwrite, secret and quota checks.
    subprocess.run([sys.executable, "-m", ENGINES[engine], *arguments], check=True)
    return {"status": "driver_exited_zero", "engine": engine,
            "quality_approved": False, "published": False}


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="two-worlds", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Offline dependency and configuration inventory")
    con = sub.add_parser("validate", help="Pydantic structural preflight (no generation)")
    con.add_argument("--kind", choices=("episode", "visual_bible", "identities"), required=True)
    con.add_argument("--input", type=Path, required=True)
    qua = sub.add_parser("quality", help="Offline release and human-review gates")
    for flag in ("video", "srt", "cover"):
        qua.add_argument("--" + flag, type=Path, required=True)
    for flag in ("episode", "assets", "original-transcript", "recognized-transcript"):
        qua.add_argument("--" + flag, type=Path)
    qua.add_argument("--full-decode", action="store_true")
    qua.add_argument("--measure-loudness", action="store_true")
    qua.add_argument("--master", action="store_true")
    qua.add_argument("--language", choices=("ru", "en"), default="ru")
    film = sub.add_parser("produce", help="Render a local film and check the decoded MP4")
    film.add_argument("--manifest", type=Path, required=True)
    film.add_argument("--output", type=Path, required=True)
    film.add_argument("--cache", type=Path, help="Private directory for resumable shot rendering")
    film.add_argument("--master", action="store_true", help="Require explicit asset approvals")
    voi = sub.add_parser("voice-plan", help="Registered English/Russian voice; default dry-run")
    voi.add_argument("--speaker", required=True)
    voi.add_argument("--language", choices=("en", "ru"), required=True)
    words = voi.add_mutually_exclusive_group(required=True)
    words.add_argument("--text")
    words.add_argument("--text-file", type=Path)
    voi.add_argument("--style", default="neutral")
    voi.add_argument("--execute", action="store_true")
    voi.add_argument("--allow-model-load", action="store_true")
    voi.add_argument("--output", type=Path)
    kaggle = sub.add_parser("kaggle-quota", help="Default offline; --live makes ONE CLI request")
    kaggle.add_argument("--live", action="store_true")
    engine = sub.add_parser("run", help="One driver interface; default plan only")
    engine.add_argument("--engine", choices=tuple(ENGINES), required=True)
    engine.add_argument("--execute", action="store_true")
    engine.add_argument("--allow-provider-usage", action="store_true")
    engine.add_argument("arguments", nargs=argparse.REMAINDER)
    return p


def main(argv: list[str] | None = None) -> int:
    p = make_parser()
    args = p.parse_args(argv)
    if args.command == "doctor":
        result = doctor()
    elif args.command == "validate":
        try:
            from studio.contracts import validate_document
        except ImportError:
            p.error("Install Pydantic first: python -m pip install -e '.[validation]'")
        result = validate_document(args.input, args.kind)
    elif args.command == "quality":
        from studio.release_gate import consolidated_check
        result = consolidated_check(
            args.video, args.srt, args.cover,
            episode=args.episode, assets=args.assets,
            original_transcript=args.original_transcript,
            recognized_transcript=args.recognized_transcript,
            full_decode=args.full_decode, master=args.master,
            measure_loudness=args.measure_loudness,
            language=args.language,
        )
    elif args.command == "produce":
        from studio.hybrid_master import produce
        from studio.production_queue import private_path

        try:
            result = produce(args.manifest, private_path(args.output), master=args.master,
                             cache=args.cache, quality_check=True)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            result = {"status": "BLOCKED", "reason": str(exc), "published": False}
    elif args.command == "voice-plan":
        from studio.voice_engine import VoiceRequest, route, execute
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        task = VoiceRequest(args.speaker, text, args.language, args.style)
        if args.execute:
            if not args.allow_model_load or args.output is None:
                p.error("Synthesis requires --execute --allow-model-load --output /private/clip.wav")
            result = execute(task, args.output)
        else:
            result = route(task)
    elif args.command == "kaggle-quota":
        if args.live:
            from studio.kaggle_api_limits import quota_once
            result = quota_once()
        else:
            result = {"status": "offline_only", "api_limit_status": "dynamic_undisclosed",
                      "gpu_quota_verified": False, "network_used": False,
                      "quota_consumed": False}
    elif args.command == "run":
        result = run_engine(args.engine, args.arguments, execute=args.execute,
                            allow_provider_usage=args.allow_provider_usage)
    else:
        p.error("Unknown command")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if str(result.get("status")).lower() == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
