"""PRIVATE V8 assembly of 30 REAL SHA-locked neural takes and six original WAVs.

No model inference, generated frames, cloned scene loops or artistic approval.
Must run only in owner-controlled PRIVATE storage; NEVER public GitHub artifacts.
An upscaled 320x576 source is not native 576x1024 neural detail.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from studio.eb01_v7_storyboard import VOICE_SHA
from studio.eb01_v8_40s_native_motion_plan import plan, verify_real_neural_sources
from studio.eb01_v8_final_mux_qa import verify_final_mux

class PrivateMovieBlocked(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def call(args: list[str], name: str, timeout: int = 240) -> bytes:
    """Never print ffmpeg stderr: it may contain PRIVATE filenames or metadata."""
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrivateMovieBlocked(f"{name}: {type(exc).__name__}") from None
    if p.returncode:
        raise PrivateMovieBlocked(f"{name}: process exit {p.returncode}")
    return p.stdout


def safe_private_path(path: Path) -> Path:
    # Check the un-resolved path first: resolve() otherwise conceals a
    # symlinked parent that can redirect private media into a public tree.
    absolute = Path(os.path.abspath(path))
    if any(node.is_symlink() for node in (absolute, *absolute.parents)):
        raise PrivateMovieBlocked("Symlinked private path or ancestor refused")
    p = absolute.resolve()
    # Also protect local runs where GITHUB_WORKSPACE is not set. A checkout
    # with .git is a publication-capable tree, never an owner media vault.
    if any((node / ".git").exists() for node in (p.parent, *p.parents)):
        raise PrivateMovieBlocked("Private media cannot reside in a Git checkout")
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace and (p == Path(workspace).resolve() or
                      Path(workspace).resolve() in p.parents):
        raise PrivateMovieBlocked("Never place private video inside PUBLIC GitHub workspace")
    # Private inputs can come from Kaggle or an owner-only temp directory.
    # Reject symlinks which could redirect original frames into public folders.
    if path.is_symlink():
        raise PrivateMovieBlocked("Symlinked private path refused")
    return p


def ffprobe(path: Path) -> dict:
    return json.loads(call(["ffprobe", "-v", "error", "-count_frames",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames,duration",
        "-of", "json", str(path)], "private ffprobe", timeout=60))


def native33(path: Path) -> None:
    p = ffprobe(path)
    streams = [s for s in p.get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1:
        raise PrivateMovieBlocked("Exactly one native source video stream required")
    v = streams[0]
    try:
        good = (v["width"] == 320 and v["height"] == 576
                and Fraction(v["avg_frame_rate"]) == 24
                and int(v["nb_read_frames"]) == 33)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        good = False
    if not good:
        raise PrivateMovieBlocked("Genuine native 320x576/24/33 source required")
    call(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-xerror",
          "-i", str(path), "-an", "-f", "null", "-"], "private source decode")


def final_frame_rgb_sha(path: Path) -> str:
    # Full final decoded 320x576 RGB frame, NOT compressed H264 file bytes.
    raw = call(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
        "-i", str(path), "-vf", "select=eq(n\\,32)", "-vsync", "0",
        "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        "source frame-32 decode", timeout=60)
    if len(raw) != 320 * 576 * 3:
        raise PrivateMovieBlocked("Missing exact final raw RGB continuity frame")
    return hashlib.sha256(raw).hexdigest()


def segment_filter(skip_conditioning_frame: bool) -> str:
    # 33 frames for first segment; only 32 for every following segment.
    # No fps converter or minterpolate: output native 24fps count unchanged.
    return ("trim=start_frame=" + ("1" if skip_conditioning_frame else "0")
            + ",setpts=PTS-STARTPTS,scale=576:1024:flags=lanczos,"
              "setsar=1,format=yuv420p")


def transcode_one(src: Path, dst: Path, skip: bool, fps=24) -> None:
    call(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
          "-i", str(src), "-an", "-vf", segment_filter(skip),
          "-fps_mode", "passthrough",
          "-frames:v", "32" if skip else "33",
          "-c:v", "libx264", "-preset", "medium", "-crf", "17",
          "-bf", "0", "-g", "24", "-pix_fmt", "yuv420p",
          str(dst)], "private neural segment transcode", timeout=250)
    p = ffprobe(dst)
    v = next((s for s in p.get("streams", []) if s.get("codec_type") == "video"), {})
    try:
        n = int(v["nb_read_frames"])
        valid = (n == (32 if skip else 33)
                 and v.get("width") == 576 and v.get("height") == 1024
                 and Fraction(v["avg_frame_rate"]) == fps)
    except (KeyError, ValueError, ZeroDivisionError):
        valid = False
    if not valid:
        raise PrivateMovieBlocked("Transcoded neural segment frame count mismatch")


def validate_private_inputs(root: Path) -> dict:
    """Hash actual GPU video bytes, test real decoding, then validate receipts."""
    schedule = plan()
    receipts = {}
    videos = []
    previous_tail_by_scene = {}
    for scene in schedule["scenes"]:
        sid = scene["scene_id"]
        rows = []
        for idx in range(scene["native_segment_count"]):
            key = f"{sid}_{idx:02d}"
            movie = root / "segments" / (key + ".mp4")
            record = root / "receipts" / (key + ".json")
            if not movie.is_file() or not record.is_file():
                raise PrivateMovieBlocked("Missing genuine private segment/receipt")
            if movie.is_symlink() or record.is_symlink() or record.stat().st_size > 32768:
                raise PrivateMovieBlocked("Unsafe segment or oversized receipt")
            receipt = json.loads(record.read_text(encoding="utf-8"))
            if (not isinstance(receipt, dict)
                    or receipt.get("mp4_sha256") != digest(movie)):
                raise PrivateMovieBlocked("Model native source SHA256 mismatch")
            native33(movie)
            tail = final_frame_rgb_sha(movie)
            if receipt.get("last_decoded_frame_sha256") != tail:
                raise PrivateMovieBlocked("Actual final decoded RGB frame hash mismatch")
            if idx > 0 and receipt.get("conditioning_input_frame_sha256") != previous_tail_by_scene.get(sid):
                raise PrivateMovieBlocked("Conditioning continuity differs from previous actual RGB frame")
            previous_tail_by_scene[sid] = tail
            rows.append(receipt)
            videos.append((sid, idx, movie))
        receipts[sid] = rows
    verified = verify_real_neural_sources(receipts)
    if verified["status"] != "SEGMENT_QA_PASS_OWNER_REVIEW_STILL_REQUIRED":
        raise PrivateMovieBlocked("Source authenticity, continuity or artist-QA receipts unverified")
    voices = {}
    for sid, sha in VOICE_SHA.items():
        wav = root / "voices" / (sid + ".wav")
        if not wav.is_file() or wav.is_symlink() or digest(wav) != sha:
            raise PrivateMovieBlocked("Canonical original Russian Qwen WAV missing")
        voices[sid] = wav
    return {"plan": schedule, "videos": videos, "voices": voices,
            "source_proof": verified}


def audio_mix(root: Path, voices: dict, schedule: dict, wav: Path) -> None:
    tracks = [s for s in schedule["scenes"] if s["scene_id"] != "S00"]
    inputs = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"]
    graph = [f"[0:a]atrim=duration={schedule['final_duration_seconds']:.6f},asetpts=PTS-STARTPTS[bed]"]
    mixed = ["[bed]"]
    for index, row in enumerate(tracks, 1):
        sid = row["scene_id"]
        start_frame = row["start_movie_frame"] + row["dialogue_start_scene_frame"]
        ms = round(start_frame * 1000 / 24)
        inputs += ["-i", str(voices[sid])]
        graph.append(f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS,adelay={ms}:all=1[s{index}]")
        mixed.append(f"[s{index}]")
    graph.append("".join(mixed) + f"amix=inputs={len(mixed)}:duration=first:normalize=0,"
                 f"atrim=duration={schedule['final_duration_seconds']:.6f},asetpts=PTS-STARTPTS[mix]")
    call(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
          *inputs, "-filter_complex", ";".join(graph),
          "-map", "[mix]", "-c:a", "pcm_s16le", str(wav)],
         "private six-line canonical audio mix", timeout=180)


def assemble_private_v8(root: Path, output: Path) -> dict:
    root = safe_private_path(root)
    output = safe_private_path(output)
    if output.exists() or output.suffix.lower() != ".mp4":
        raise PrivateMovieBlocked("Refuse to overwrite private media / invalid MP4")
    if not (root.is_dir() and (root / "segments").is_dir()
            and (root / "receipts").is_dir() and (root / "voices").is_dir()):
        raise PrivateMovieBlocked("Private original film source directories absent")
    proof = validate_private_inputs(root)
    movie = proof["plan"]
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.parent.is_symlink():
        raise PrivateMovieBlocked("Symlink destination is unsafe")
    # Temporary original media is never written to a public GitHub artifact.
    with tempfile.TemporaryDirectory(prefix="eb01-private-v8-", dir=str(output.parent)) as temp:
        workspace = Path(temp)
        encoded = []
        for serial, (sid, idx, input_mp4) in enumerate(proof["videos"]):
            candidate = workspace / f"{serial:02d}.mp4"
            transcode_one(input_mp4, candidate, skip=idx > 0)
            encoded.append(candidate)
        playlist = workspace / "private_takes.ffconcat"
        playlist.write_text("ffconcat version 1.0\n"
            + "".join("file '" + p.name + "'\n" for p in encoded), encoding="utf-8")
        joined = workspace / "joined_private_30_native_takes.mp4"
        call(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
             "-safe", "0", "-f", "concat", "-i", str(playlist),
             "-c:v", "copy", "-an", str(joined)],
             "private thirty-source concat", timeout=300)
        p = ffprobe(joined)
        v = next((s for s in p.get("streams", []) if s.get("codec_type") == "video"), {})
        if int(v.get("nb_read_frames", "-1")) != movie["final_unique_native_frames"]:
            raise PrivateMovieBlocked("30 segments missing expected 967 original frames")
        mixed = workspace / "six_original_russian_lines.wav"
        audio_mix(root, proof["voices"], movie, mixed)
        candidate = workspace / "private_owner_40s_review_unapproved.mp4"
        call(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
              "-i", str(joined), "-i", str(mixed),
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
              "-c:a", "aac", "-b:a", "192k",
              "-movflags", "+faststart", str(candidate)],
             "private six-line mux", timeout=300)
        info = ffprobe(candidate)
        v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
        n = int(v.get("nb_read_frames", "-1"))
        decision = verify_final_mux(
            info, decoded_video_frames=n,
            independent_scene_count=7,
            canonical_audio_sha_verified=True,
            original_neural_segments_verified=True,
            six_line_lipsync_verified=False,
            owner_final_artistic_review=False,
            private_durable_storage_verified=False)
        if n != 967 or "AUDIO_STREAM_DURATION_MISSING" in decision["reasons"]:
            raise PrivateMovieBlocked("Private review mux clock or voice track incomplete")
        # Preserve unapproved review media privately. No 'final approved' claim.
        shutil.move(str(candidate), str(output))
    return {"schema": 8,
            "status": "PRIVATE_ORIGINAL_40S_REVIEW_MP4_NOT_APPROVED",
            "original_video_sha256": digest(output),
            "video_frame_count": 967, "seconds": movie["final_duration_seconds"],
            "original_neural_segments": 30, "independent_scenes": 7,
            "canonical_russian_voice_lines": 6,
            "scaled_from_native_320x576": True,
            "native_576x1024_detail_verified": False,
            "audio_conditioned_lipsync_verified": False,
            "owner_artistic_approval": False,
            "private_durable_storage_verified": False,
            "public_gitHub_artifact_export": False,
            "full_approved_movie_ready": False,
            "final_quality_gate": decision}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--private-source-root", type=Path, required=True)
    p.add_argument("--private-review-output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(assemble_private_v8(args.private_source_root, args.private_review_output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
