"""Offline hybrid short-video editor: stills + optional pre-generated motion clips.

No cloud service, model download, automatic publishing or fabricated approvals.
JSON shot list is authored externally (e.g. approved by Work). FFmpeg does
camera motion, stitches shots, aligns dialogue, mixes optional soundtrack,
and exports one MP4 plus separate SRT and a reproducible QA receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image

from studio.smooth_camera import CAMERAS, encode_rgb_frames, render_still
from studio.release_gate import sha256
from studio.video_qc import inspect_temporal, validate_regions
IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
CLIPS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SHOT_TYPES = frozenset({"establishing", "wide", "medium", "close", "extreme_close",
                        "over_shoulder", "reaction", "insert"})
VISUAL_PROFILES = {"photoreal_cinematic", "stylized_cinematic", "3d_character"}
CLIP_ENGINES = {"blender_3d", "neural_character"}


def source(base: Path, rel: str, suffixes: set[str]) -> Path:
    """Explicitly relative, in-project sources; prohibit path escapes/symlinks."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise ValueError("Media paths must be nonempty and relative to the manifest")
    root = base.resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"Missing or out-of-project asset: {rel}")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported media type: {rel}")
    return path


def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,duration,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    return json.loads(proc.stdout)


def has_stream(path: Path, kind: str) -> bool:
    return any(s.get("codec_type") == kind for s in probe(path).get("streams", []))


def sound_stems(data: dict, base: Path, duration: float, *, master: bool) -> list[dict]:
    from studio.audio_master import RIGHTS, TIMELINE_BUSES, describe_audio

    raw = data.get("sound_stems", [])
    if not isinstance(raw, list) or len(raw) > 32:
        raise ValueError("Use up to 32 timed sound stems")
    stems = []
    for item in raw:
        if (not isinstance(item, dict) or not isinstance(item.get("bus"), str)
                or item["bus"] not in TIMELINE_BUSES):
            raise ValueError("Invalid sound stem bus")
        start, gain, loop = item.get("start", 0), item.get("gain", 1), item.get("loop", False)
        if (type(start) not in (int, float) or not 0 <= start < duration
                or type(gain) not in (int, float) or not 0 < gain <= 2
                or type(loop) is not bool
                or loop and item["bus"] not in {"music", "ambience"}):
            raise ValueError("Invalid sound stem timing, gain or loop")
        if master and (item.get("review_status") != "approved" or item.get("rights") not in RIGHTS
                       or len(str(item.get("rights_note", "")).strip()) < 8):
            raise ValueError("Master sound stems need review and documented rights")
        path = source(base, item.get("path"), AUDIO)
        info = describe_audio(path)
        if item.get("sha256", info["sha256"]) != info["sha256"]:
            raise ValueError("Sound stem SHA-256 mismatch")
        if item["bus"] in {"dialogue", "narration"}:
            raise ValueError("Time speech through its shot voice so captions and lips stay aligned")
        stems.append({"bus": item["bus"], "path": path, "start": start, "gain": gain,
                      "loop": loop, "duration": duration - start if loop else info["duration"]})
    return stems


def shot_regions(item: dict, duration: float, fps: int) -> list[dict]:
    regions = item.get("protected_regions", [])
    if not isinstance(regions, list) or len(regions) > 24:
        raise ValueError("Invalid protected regions list")
    for region in regions:
        if not isinstance(region, dict) or set(region) != {"id", "layer", "box", "start", "end"}:
            raise ValueError("A protected region needs an ID, layer, box and visibility window")
        start, end, box = region["start"], region["end"], region["box"]
        if (not isinstance(region["id"], str) or not 1 <= len(region["id"]) <= 100
                or type(region["layer"]) is not int or region["layer"] < 0
                or not isinstance(box, list) or len(box) != 4
                or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in box)
                or box[0] >= box[2] or box[1] >= box[3]
                or any(type(x) not in (int, float) for x in (start, end))
                or not 0 <= start < end <= duration
                or math.ceil(start * fps - 1e-7) >= math.ceil(end * fps - 1e-7)):
            raise ValueError("Invalid protected box or visibility window")
    return regions


def animation_engine(kind: str, cinematic: dict | None, take: dict | None) -> str:
    if kind == "clip":
        return take["engine"] if take is not None else "imported_clip"
    if kind == "i2v":
        return "neural_i2v"
    if kind == "image":
        return "still_camera"
    layers = cinematic["layers"]
    if any("performance_take" in layer.get("facial", {}) for layer in layers):
        return "registered_character_performance"
    if any({"rig", "deform", "motion"} & layer.keys() for layer in layers):
        return "character_rig"
    return "cinematic_motion_comic"


def check_manifest(data: dict, base: Path, *, master: bool = False) -> dict:
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("Expected hybrid shot manifest with schema=1")
    width, height, fps = data.get("width", 1080), data.get("height", 1920), data.get("fps", 24)
    if not all(type(i) is int for i in (width, height, fps)):
        raise ValueError("Resolution and fps must be integers")
    if not (width >= 288 and height >= 512 and width <= 2160 and height <= 3840
            and width % 2 == 0 and height % 2 == 0 and abs(width / height - 9 / 16) < .001
            and 12 <= fps <= 30):
        raise ValueError("Expected an even 9:16 resolution and 12-30 fps")
    burn_subtitles = data.get("burn_subtitles", False)
    if type(burn_subtitles) is not bool:
        raise ValueError("burn_subtitles must be a boolean")
    language = data.get("language")
    if language not in (None, "ru", "en"):
        raise ValueError("Subtitles support ru or en")
    profile = data.get("visual_profile")
    if profile is not None and (not isinstance(profile, str) or profile not in VISUAL_PROFILES):
        raise ValueError("Unknown visual profile")
    shots = data.get("shots")
    if not isinstance(shots, list) or not (1 <= len(shots) <= 80):
        raise ValueError("Provide 1-80 shots")
    if master and not (data.get("editorial_approved") is True and
                       data.get("rights_approved") is True):
        raise ValueError("Master requires explicit editorial and rights approvals")
    normalized, start = [], 0
    for i, item in enumerate(shots):
        if not isinstance(item, dict):
            raise ValueError(f"Shot {i} is not an object")
        duration = item.get("duration")
        if type(duration) not in (float, int) or not math.isfinite(duration) or not 0.5 <= duration <= 30:
            raise ValueError(f"Shot {i} duration must be 0.5-30 seconds")
        frames = round(duration * fps)
        if frames < 1:
            raise ValueError(f"Shot {i} is too short")
        kind = "clip" if "clip" in item else "image" if "image" in item else None
        if kind is None or ("clip" in item and "image" in item):
            raise ValueError(f"Shot {i} needs exactly one image or clip")
        media = source(base, item[kind], CLIPS if kind == "clip" else IMAGES)
        take = item.get("clip_take")
        video_stream = None
        if kind == "clip":
            video_stream = next((s for s in probe(media).get("streams", []) if s["codec_type"] == "video"), None)
            if video_stream is None:
                raise ValueError(f"Shot {i}: clip has no video stream")
        if take is not None and (kind != "clip" or not isinstance(take, dict)
                or set(take) != {"engine", "visual_profile", "sha256"}
                or not isinstance(take["engine"], str) or take["engine"] not in CLIP_ENGINES
                or profile is None or take["visual_profile"] != profile or sha256(media) != take["sha256"]):
            raise ValueError(f"Shot {i}: clip take must match the locked source and visual profile")
        if kind == "clip" and profile is not None and take is None:
            raise ValueError(f"Shot {i}: visual profile requires a source-pinned clip take")
        camera = item.get("camera", "push_in")
        if camera not in CAMERAS:
            raise ValueError(f"Shot {i}: unknown camera move {camera}")
        strength = item.get("camera_strength", 0.45)
        if type(strength) not in (float, int) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError(f"Shot {i}: camera_strength must be between 0 and 1")
        cinematic = item.get("cinematic")
        i2v = item.get("i2v")
        if ("cinematic" in item and not isinstance(cinematic, dict)
                or (cinematic is not None or i2v is not None) and kind != "image"):
            raise ValueError("Layered and I2V shots require their reference image")
        backend = item.get("animation_backend", "auto")
        if backend not in ("auto", "rig", "i2v") or (backend == "rig" and cinematic is None):
            raise ValueError("Animation backend must match the supplied shot assets")
        shot_type = item.get("shot_type", "medium")
        position = item.get("caption_position", "bottom")
        if not isinstance(shot_type, str) or shot_type not in SHOT_TYPES or position not in ("top", "bottom"):
            raise ValueError("Unknown shot type or caption position")
        regions = shot_regions(item, frames / fps, fps)
        if regions and (cinematic is None or backend == "i2v"):
            raise ValueError("Protected regions require a tracked layered composition")
        qc_regions = (validate_regions(item["qc_regions"], (width, height), duration=frames / fps)
                      if "qc_regions" in item else [])
        if any("identity_reference" in region for region in qc_regions):
            from studio.identity_qc import locked_region_path

            for region in qc_regions:
                if "identity_reference" in region:
                    locked_region_path(region["identity_reference"], base)
        strict_clip = take is not None or any(r["motion_required"] or r.get("motion_windows") for r in qc_regions)
        if video_stream is not None and strict_clip:
            length = video_stream.get("duration")
            if length not in (None, "N/A") and float(length) + 1e-4 < frames / fps:
                raise ValueError(f"Shot {i}: source clip is shorter than the requested action")
        fade = item.get("fade_out", 0.)
        if (type(fade) not in (int, float) or not 0 <= fade <= min(1.5, frames / fps / 2)
                or fade and (cinematic is None or i != len(shots) - 1 or
                             any(r["end"] > frames / fps - fade for r in regions))):
            raise ValueError("Use a bounded fade only on the final layered shot and outside protected windows")
        caption = item.get("caption", "")
        if not isinstance(caption, str) or len(caption) > 240:
            raise ValueError(f"Shot {i}: caption too long")
        # Optional real, per-phrase timing supplied by an editor/aligner.
        # Never infer word positions from the caption or invent speech timing.
        subtitle_cues = item.get("subtitle_cues", [])
        if not isinstance(subtitle_cues, list) or len(subtitle_cues) > 40:
            raise ValueError(f"Shot {i}: invalid subtitle_cues list")
        validated_cues = []
        previous_cue_end = 0.0
        for cue in subtitle_cues:
            if not isinstance(cue, dict):
                raise ValueError(f"Shot {i}: subtitle cue must be an object")
            begin, end, txt = cue.get("start"), cue.get("end"), cue.get("text")
            if (type(begin) not in (float, int) or type(end) not in (float, int)
                or not math.isfinite(begin) or not math.isfinite(end)
                or begin < previous_cue_end or end <= begin
                or end > frames / fps + 0.00001
                or not isinstance(txt, str) or not txt.strip() or len(txt) > 120):
                raise ValueError(f"Shot {i}: invalid/overlapping subtitle cue timing")
            validated_cues.append({"start": float(begin), "end": float(end), "text": txt.strip()})
            previous_cue_end = end
        approved = item.get("approved") is True
        if master and not approved:
            raise ValueError(f"Shot {i}: asset not explicitly approved")
        voice = None
        voice_offset = item.get("voice_offset", 0)
        voice_bus = item.get("voice_bus", "dialogue")
        cleanup = item.get("voice_cleanup", False)
        if (type(voice_offset) not in (int, float) or not 0 <= voice_offset < frames / fps
                or voice_bus not in ("dialogue", "narration") or type(cleanup) is not bool):
            raise ValueError("Invalid voice offset, bus or cleanup option")
        voice_offset = round(voice_offset * 1000) / 1000
        voice_duration = 0.
        if item.get("voice"):
            voice = source(base, item["voice"], AUDIO)
            if not has_stream(voice, "audio"):
                raise ValueError(f"Shot {i}: no audio in voice track")
            voice_duration = float(probe(voice).get("format", {}).get("duration", 0))
            padding = 0.003 if cinematic is not None or "voice_offset" in item or "sound_stems" in data else .15
            if not 0 < voice_duration <= 600 or voice_duration + voice_offset > frames / fps + padding:
                raise ValueError(f"Shot {i}: voice exceeds shot duration; extend the shot to fit the recording")
            if master and item.get("voice_approved") is not True:
                raise ValueError(f"Shot {i}: voice must be explicitly approved")
        if cinematic is not None:
            from studio.premium_assets import validate_premium_scene
            validate_premium_scene({"cinematic": cinematic}, base, publish=master)
        if "dialogue_text" in item:
            locked = item["dialogue_text"]
            cue_text = " ".join(c["text"] for c in validated_cues) or caption
            if (not isinstance(locked, str) or not voice or not locked.strip()
                    or locked.split() != cue_text.split()):
                raise ValueError("Subtitle words do not match the locked dialogue")
        fallback = None
        reference = media
        motion_receipt = None
        preserve_portrait = (backend == "auto" and profile == "photoreal_cinematic" and cinematic is not None
                             and any(layer.get("role") == "character" and layer.get("sha256")
                                     and ("rig" in layer or "deform" in layer)
                                     for layer in cinematic["layers"]))
        if backend != "rig" and i2v is not None and not regions and not preserve_portrait:
            from studio.neural_motion import import_take
            try:
                if not isinstance(i2v, dict) or set(i2v) != {"clip", "receipt"}:
                    raise ValueError("I2V requires a clip and its worker receipt")
                candidate = source(base, i2v["clip"], CLIPS)
                proof = source(base, i2v["receipt"], {".json"})
                motion_receipt = import_take(candidate, proof, reference, frames / fps)
                media = candidate
                kind = "i2v"
            except (ValueError, OSError, subprocess.SubprocessError):
                if backend == "i2v" or cinematic is None:
                    raise ValueError(f"Shot {i}: I2V needs a usable, source-pinned video take") from None
                fallback = "i2v_take_missing_or_invalid"
        elif backend == "i2v":
            raise ValueError("Explicit I2V needs a video take")
        elif i2v is not None and regions:
            fallback = "i2v_excluded_to_preserve_protected_region"
        elif i2v is not None and preserve_portrait:
            fallback = "i2v_excluded_to_preserve_locked_portrait"
        if cinematic is not None and kind != "i2v":
            kind = "layered"
        engine = animation_engine(kind, cinematic, take)
        if profile == "3d_character" and (
                engine in {"character_rig", "neural_i2v", "registered_character_performance", "neural_character"}
                or shot_type in {"medium", "close", "extreme_close", "over_shoulder", "reaction"}
                and engine != "blender_3d"):
            raise ValueError(f"Shot {i}: 3D character profile requires a locked Blender take for character plans")
        if fade and kind != "layered":
            raise ValueError("A layered fade requires the layered renderer")
        normalized.append(dict(kind=kind, path=media, frames=frames,
                               duration=frames / fps, start=start / fps, camera=camera, strength=strength,
                               caption=caption.strip(), voice=voice, approved=approved,
                               subtitle_cues=validated_cues, voice_offset=voice_offset,
                               dialogue_text=item.get("dialogue_text"),
                               voice_duration=voice_duration, voice_bus=voice_bus, voice_cleanup=cleanup,
                               shot_type=shot_type, caption_position=position, cinematic=cinematic,
                               protected_regions=regions, reference=reference, fallback=fallback,
                               qc_regions=qc_regions, strict_clip=strict_clip or kind == "i2v",
                               visual_profile=profile, animation_engine=engine, fade_out=fade,
                               motion_receipt=motion_receipt, base=base.resolve()))
        start += frames
    if start / fps > 600:
        raise ValueError("Video longer than 10 minutes")
    music = source(base, data["music"], AUDIO) if data.get("music") else None
    if music is not None and not has_stream(music, "audio"):
        raise ValueError("Music source has no audio")
    if master and music and data.get("music_approved") is not True:
        raise ValueError("Master soundtrack must be explicitly approved")
    volume = data.get("music_volume", .16)
    if type(volume) not in (int, float) or not math.isfinite(volume) or not 0 <= volume <= .5:
        raise ValueError("music_volume must be between 0 and 0.5")
    required_clues = data.get("required_clues", [])
    if (not isinstance(required_clues, list) or any(not isinstance(v, str) for v in required_clues)
            or not set(required_clues) <= {r["id"] for s in normalized for r in s["protected_regions"]}):
        raise ValueError("Required clue is missing a protected visibility window")
    stems = sound_stems(data, base, start / fps, master=master)
    upgraded = ("sound_stems" in data or bool(required_clues) or any(
        shot["kind"] in {"layered", "i2v"} or shot["voice_offset"]
        or shot["voice_cleanup"] or shot["voice_bus"] != "dialogue" or profile is not None
        or shot["caption_position"] != "bottom" for shot in normalized))
    return dict(width=width, height=height, fps=fps, shots=normalized,
                frames=start, duration=start / fps, music=music, music_volume=volume,
                approved=master, burn_subtitles=burn_subtitles, sound_stems=stems,
                upgraded=upgraded, required_clues=required_clues, language=language, visual_profile=profile)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(command, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, timeout=1200, cwd=cwd)
    if result.returncode:
        raise RuntimeError("FFmpeg export failed: " + result.stderr.decode(errors="replace")[-1400:])


def cue_rows(shot: dict) -> list[dict]:
    if shot["subtitle_cues"]:
        return shot["subtitle_cues"]
    if shot["caption"]:
        return [{"start": 0., "end": shot["duration"], "text": shot["caption"]}]
    return []


def render_segment(shot: dict, path: Path, width: int, height: int, fps: int, *,
                   master: bool = False, burn_subtitles: bool = False) -> dict:
    if shot["kind"] == "layered":
        from studio.cinematic import CinematicPanel

        scene = {"cinematic": shot["cinematic"], "duration": shot["duration"],
                 "protected_regions": shot["protected_regions"]}
        if shot["voice"]:
            scene["voice"] = {"path": shot["voice"].relative_to(shot["base"]).as_posix(),
                              "offset_ms": round(shot["voice_offset"] * 1000),
                              "text": shot.get("dialogue_text")}
        if burn_subtitles:
            zone = (.06, .26) if shot["caption_position"] == "top" else (.64, .855)
            scene["caption_windows"] = [(c["start"], c["end"], *zone) for c in cue_rows(shot)]
        panel = CinematicPanel(scene, shot["base"], width, height, master=master, fps=fps)
        if shot["visual_profile"] == "photoreal_cinematic" and (
                "amplitude_only_not_phonemes" in panel.performance_modes or panel.voice_levels is not None):
            raise ValueError("Photoreal dialogue requires a timed performance or an editorial reaction shot")
        if shot["fallback"] == "i2v_take_missing_or_invalid" and not panel.active_actors:
            raise ValueError("I2V fallback requires an articulated or deformable actor")
        frames = shot["frames"]
        def pictures():
            matte = Image.new("RGB", (width, height), (17, 19, 22)) if shot["fade_out"] else None
            for i in range(frames):
                frame = panel.exposed_frame(i, frames, fps)
                if matte is not None:
                    amount = min(1., (frames - 1 - i) / (shot["fade_out"] * fps))
                    if amount < 1:
                        frame = Image.blend(matte, frame, amount * amount * (3 - 2 * amount))
                yield frame

        encode_rgb_frames(path, width, height, fps, frames, pictures())
        return {"active_actor_layers": panel.active_actors,
                "lipsync_modes": panel.performance_modes,
                "protected_windows_checked": len(panel.regions),
                **panel.optics_report()}
    if shot["kind"] == "image":
        render_still(shot["path"], path, width, height, fps,
                     shot["frames"], shot["camera"], shot["strength"])
        return {}
    inputs = ["-i", str(shot["path"])]
    filters = (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
               f"crop={width}:{height},setsar=1,setpts=PTS-STARTPTS,fps={fps},setpts=N/{fps}/TB")
    if not shot["strict_clip"]:
        filters += f",tpad=stop_mode=clone:stop_duration={shot['duration']}"
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
         *inputs, "-vf", filters + ",format=yuv420p", "-frames:v", str(shot["frames"]),
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
         "-pix_fmt", "yuv420p", "-r", str(fps), str(path)])
    if shot["strict_clip"]:
        stream = next((s for s in probe(path).get("streams", []) if s["codec_type"] == "video"), {})
        if stream.get("nb_frames") != str(shot["frames"]):
            raise ValueError("Clip did not provide its full action without holding a frame")
    return {"frame_padding_allowed": not shot["strict_clip"]}


def timestamp(t: float) -> str:
    ms = round(t * 1000)
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def subtitles(shots: list[dict]) -> str:
    """Prefer explicit measured phrase cues, otherwise one whole-shot caption."""
    rows = []
    for shot in shots:
        cues = shot.get("subtitle_cues") or []
        if cues:
            for cue in cues:
                rows.append((shot["start"] + cue["start"],
                             shot["start"] + cue["end"], cue["text"]))
        elif shot["caption"]:
            rows.append((shot["start"], shot["start"] + shot["duration"], shot["caption"]))
    result = []
    for number, (start, end, line) in enumerate(rows, 1):
        result.append(f"{number}\n{timestamp(start)} --> {timestamp(end)}\n{line}\n")
    return "\n".join(result) + ("\n" if result else "")



def _ass_time(t: float) -> str:
    # ASS supports centiseconds; video rendering is frame-accurate at fps.
    centiseconds = round(t * 100)
    h, r = divmod(centiseconds, 360000)
    m, r = divmod(r, 6000)
    sec, cs = divmod(r, 100)
    return f"{h}:{m:02}:{sec:02}.{cs:02}"


def _ass_text(value: str) -> str:
    # ASS override tags, if present in source text, must not execute.
    return (value.replace(chr(92), "＼")
                 .replace("{", "｛").replace("}", "｝")
                 .replace("\n", r"\N"))


def ass_subtitles(shots: list[dict], width: int, height: int) -> str:
    """Create a TikTok-safe Cyrillic-capable caption track above UI overlays."""
    font = max(22, min(76, round(width * .056)))
    margin = max(70, round(height * .165))
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Story,DejaVu Sans,{font},&H00FFFFFF,&H0000FFFF,"
        f"&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,"
        f"{max(22,width//13)},{max(22,width//13)},{margin},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for shot in shots:
        cues = shot.get("subtitle_cues") or []
        if not cues and shot.get("caption"):
            cues = [{"start": 0.0, "end": shot["duration"], "text": shot["caption"]}]
        for cue in cues:
            start, end = shot["start"] + cue["start"], shot["start"] + cue["end"]
            text = _ass_text(cue["text"])
            if shot.get("measured_captions"):
                from studio.caption_layout import layout

                readable = layout(cue["text"], preferred_size=font, minimum_size=max(16, round(font * .82)),
                                  max_width=width - 2 * max(22, width // 13) - 8)
                text = r"{\q2\fs" + str(readable["font_size"]) + "}" + r"\N".join(
                    _ass_text(line) for line in readable["lines"])
            vertical = 0
            if shot.get("caption_position") == "top":
                text = r"{\an8}" + text
                vertical = round(height * .085)
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                f"Story,,0,0,{vertical},,{text}"
            )
    return header + "\n".join(events) + ("\n" if events else "")


def burn_ass_captions(source: Path, dest: Path, plan: dict, *, folder: Path) -> None:
    """Render large text into video frames; preserve the existing AAC audio."""
    if not any(s["caption"] or s.get("subtitle_cues") for s in plan["shots"]):
        raise ValueError("Cannot burn an empty subtitle track")
    path = folder / "captions.ass"
    path.write_text(ass_subtitles(plan["shots"], plan["width"], plan["height"]),
                    encoding="utf-8")
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", "ass=captions.ass",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-r", str(plan["fps"]),
        "-c:a", "copy", "-movflags", "+faststart", str(dest)
    ], cwd=folder)


def add_audio(video: Path, output: Path, plan: dict) -> dict:
    """Mix genuine speech with an automatically ducked licensed music bed."""
    length = plan["duration"]
    if plan["upgraded"]:
        from studio.audio_master import mix_timeline
        from studio.delivery_qc import MAX_AAC_TRUE_PEAK_DBTP, loudness

        tracks = list(plan["sound_stems"])
        if plan["music"] and plan["music_volume"] > 0:
            tracks.append({"bus": "music", "path": plan["music"], "start": 0,
                           "duration": length, "gain": plan["music_volume"], "loop": True})
        for shot in plan["shots"]:
            if shot["voice"]:
                tracks.append({"bus": shot["voice_bus"], "path": shot["voice"],
                               "start": shot["start"] + shot["voice_offset"],
                               "duration": shot["voice_duration"], "gain": 1.,
                               "cleanup": shot["voice_cleanup"]})
        if tracks:
            wav = output.with_suffix(".wav")
            receipt = mix_timeline(tracks, length, wav)

            def mux(audio_filters: list[str]) -> None:
                run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-i", str(wav),
                     "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                     "-b:a", "192k", "-ar", "48000", *audio_filters, "-t", f"{length:.8f}",
                     "-movflags", "+faststart", str(output)])

            mux([])
            peak = loudness(output)["true_peak_dbtp"]
            if peak > MAX_AAC_TRUE_PEAK_DBTP:
                # AAC reconstruction can exceed the true peak of its source PCM.
                # Encode from that PCM again with room for a second AAC overshoot.
                reduction = round(peak - MAX_AAC_TRUE_PEAK_DBTP + .5, 2)
                mux(["-af", f"volume=-{reduction:.2f}dB"])
                receipt["aac_peak_correction_db"] = reduction
            return receipt
    inputs = ["-i", str(video)]
    music_index = None
    voices = []
    if plan["music"]:
        music_index = 1
        inputs.extend(["-stream_loop", "-1", "-i", str(plan["music"])])
    for shot in plan["shots"]:
        if shot["voice"]:
            # Count input descriptors, not CLI argument list items.
            index = 1 + (music_index is not None) + len(voices)
            inputs.extend(["-i", str(shot["voice"])])
            voices.append((index, shot))
    filters = []
    if music_index is not None:
        filters.append(
            f"[{music_index}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={plan['music_volume']},apad,atrim=duration={length},"
            "asetpts=PTS-STARTPTS[bed]"
        )
    for i, (index, shot) in enumerate(voices):
        delay = round(shot["start"] * 1000)
        filters.append(
            f"[{index}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"atrim=duration={shot['duration']},asetpts=PTS-STARTPTS,"
            f"adelay={delay}:all=1,apad,atrim=duration={length}[v{i}]"
        )
    if voices:
        voice_labels = "".join(f"[v{i}]" for i in range(len(voices)))
        if len(voices) == 1:
            filters.append(f"{voice_labels}anull[voice]")
        else:
            filters.append(f"{voice_labels}amix=inputs={len(voices)}:"
                           f"duration=longest:normalize=0[voice]")
        if music_index is not None:
            filters.append("[voice]asplit=2[voice_main][voice_key]")
            filters.append(
                "[bed][voice_key]sidechaincompress=threshold=0.04:"
                "ratio=6:attack=25:release=380[ducked]"
            )
            filters.append(
                f"[voice_main][ducked]amix=inputs=2:duration=longest:"
                f"normalize=0,atrim=duration={length},"
                "alimiter=limit=0.90[aout]"
            )
        else:
            filters.append(
                f"[voice]atrim=duration={length},alimiter=limit=0.90[aout]"
            )
    elif music_index is not None:
        filters.append("[bed]alimiter=limit=0.90[aout]")
    else:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        filters.append(
            f"[1:a]atrim=duration={length},asetpts=PTS-STARTPTS[aout]"
        )
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        *inputs, "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", f"{length:.5f}", "-movflags", "+faststart", str(output)
    ])
    return {}


def shot_sources(shot: dict) -> set[Path]:
    from studio.performance_take import take_sources
    from studio.premium_assets import resolve_asset

    paths = {shot["path"], shot["reference"]}
    if shot["voice"]:
        paths.add(shot["voice"])
    if shot["kind"] == "layered":
        for layer in shot["cinematic"]["layers"]:
            paths.add(resolve_asset(shot["base"], layer["path"]))
            if "rig" in layer:
                paths.update(resolve_asset(shot["base"], part["path"]) for part in layer["rig"]["parts"])
            for category in ("eyes", "mouth"):
                paths.update(resolve_asset(shot["base"], plate["path"])
                             for plate in layer.get("facial", {}).get(category, {}).values())
            for category in ("blink", "mouth"):
                if category in layer.get("motion", {}):
                    paths.add(resolve_asset(shot["base"], layer["motion"][category]["path"]))
            for timing in ("viseme_track", "phoneme_track"):
                if timing in layer.get("facial", {}):
                    paths.add(source(shot["base"], layer["facial"][timing], {".json"}))
            if "performance_take" in layer.get("facial", {}):
                paths.update(take_sources(shot["base"], layer["facial"]["performance_take"]))
    return paths


@lru_cache(maxsize=1)
def renderer_version() -> list[str]:
    from numpy import __version__ as numpy_version
    from PIL import __version__ as pillow_version

    ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=10)
    modules = ("hybrid_master", "cinematic", "smooth_camera", "actor_motion", "character_engine",
               "cinematic_lighting", "deformable_actor", "facial_actor", "lipsync", "performance_take")
    return [pillow_version, numpy_version, ffmpeg_version.splitlines()[0], *(
        sha256(Path(__file__).with_name(name + ".py")) for name in modules
    )]


def shot_key(shot: dict, source_shot: dict, plan: dict) -> str:
    paths = shot_sources(shot)
    data = {"renderer": renderer_version(), "dimensions": [plan["width"], plan["height"], plan["fps"]],
            "burn_subtitles": plan["burn_subtitles"], "master": plan["approved"], "shot": source_shot,
            "backend": shot["kind"], "visual_profile": shot["visual_profile"],
            "strict_clip": shot["strict_clip"], "sources": sorted(
                (p.relative_to(shot["base"]).as_posix(), sha256(p)) for p in paths)}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def produce(manifest: Path, output: Path, *, master: bool = False, cache: Path | None = None,
            quality_check: bool = False) -> dict:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and ffprobe must be installed")
    manifest = manifest.resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    plan = check_manifest(data, manifest.parent, master=master)
    output = output.resolve()
    if output.suffix.lower() != ".mp4" or output == manifest:
        raise ValueError("Output must be a new .mp4 file")
    outputs = [output.with_suffix(".srt"), output.with_suffix(".json"), output]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("Output MP4 or sidecar already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tw-hybrid-", dir=output.parent) as td:
        temp = Path(td)
        segments, shot_reports = [], []
        for n, shot in enumerate(plan["shots"]):
            shot["measured_captions"] = plan["upgraded"]
            segment = temp / f"shot_{n:03d}.mp4"

            def render(path):
                return render_segment(shot, path, plan["width"], plan["height"], plan["fps"],
                                      master=master, burn_subtitles=plan["burn_subtitles"])

            if cache is None:
                stats, reused = render(segment), False
            else:
                from studio.model_cache import cached_segment

                fingerprint = shot_key(shot, data["shots"][n], plan)

                def render_unchanged(path):
                    result = render(path)
                    if shot_key(shot, data["shots"][n], plan) != fingerprint:
                        raise ValueError("Shot source changed during rendering")
                    return result

                stats, reused = cached_segment(cache, fingerprint, segment, render_unchanged)
            if shot["qc_regions"]:
                speech = None
                if shot["voice"] and any(r["kind"] == "mouth" for r in shot["qc_regions"]):
                    from studio.lipsync import speech_activity

                    speech = speech_activity(shot["voice"])
                temporal = inspect_temporal(segment, shot["qc_regions"], plan["width"],
                                            plan["height"], plan["fps"], shot["frames"],
                                            speech=speech, voice_offset=shot["voice_offset"],
                                            assets=shot["base"])
                flow = temporal.get("optical_flow", {})
                groups = temporal["regions"] + flow.get("regions", [])
                if flow.get("events"):
                    groups.append({"id": "optical-flow", "events": flow["events"]})
                for region in groups:
                    for event in region["events"]:
                        event["timeline_start_seconds"] = round(shot["start"] + event["start_seconds"], 4)
                        event["timeline_end_seconds"] = round(shot["start"] + event["end_seconds"], 4)
                        if event["severity"] == "BLOCKED":
                            raise ValueError(
                                f"BLOCKED: shot {n + 1}, {region['id']}, {event['code']}, "
                                f"{event['start_seconds']:.3f}–{event['end_seconds']:.3f}s"
                            )
                stats = {**stats, "temporal_qc": temporal}
            shot_reports.append({"shot": n + 1, "shot_type": shot["shot_type"], "backend": shot["kind"],
                                 "frames": shot["frames"], "cached": reused, "start_seconds": shot["start"],
                                 "animation_engine": shot["animation_engine"],
                                 "fallback": shot["fallback"], "motion_source": shot["motion_receipt"],
                                 **stats})
            segments.append(segment)
        concat = temp / "concat.txt"
        concat.write_text("".join(f"file '{p.name}'\nduration {s['duration']:.9f}\n"
                                  for p, s in zip(segments, plan["shots"])), encoding="utf-8")
        silent = temp / "silent.mp4"
        run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "1", "-i", str(concat),
             "-c:v", "copy", str(silent)])
        staged = temp / "mixed.mp4"
        mix_report = add_audio(silent, staged, plan)
        if plan["burn_subtitles"]:
            burnt = temp / "burned_in.mp4"
            burn_ass_captions(staged, burnt, plan, folder=temp)
            staged = burnt
        measured = probe(staged)
        streams = measured.get("streams", [])
        videos = [s for s in streams if s.get("codec_type") == "video"]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        seconds = float(measured.get("format", {}).get("duration", 0))
        if (len(videos) != 1 or len(audios) != 1 or
                videos[0].get("width") != plan["width"] or
                videos[0].get("height") != plan["height"] or
                abs(seconds - plan["duration"]) > .2):
            raise RuntimeError("Export failed video/audio/duration quality checks")
        srt = temp / "out.srt"
        srt.write_text(subtitles(plan["shots"]), encoding="utf-8")
        qc = None
        if plan["upgraded"] or quality_check or any("temporal_qc" in s for s in shot_reports):
            from studio.quality_hub import assess_cinematic

            qc = assess_cinematic(staged, srt, plan)
            if qc["status"] == "BLOCKED":
                raise RuntimeError("BLOCKED: " + qc["reason"])
            if any(s.get("temporal_qc", {}).get("status") == "REVIEW REQUIRED" for s in shot_reports):
                qc["review_required"].append("temporal_artifacts")
        receipt = {
            "schema": 1, "source_manifest": manifest.name, "source_sha256": sha256(manifest),
            "video_sha256": sha256(staged), "resolution": [plan["width"], plan["height"]],
            "fps": plan["fps"], "duration_seconds": round(seconds, 3), "shots": len(plan["shots"]),
            "motion_clips": sum(s["kind"] in {"clip", "i2v"} for s in plan["shots"]),
            "static_camera_shots": sum(s["kind"] == "image" for s in plan["shots"]),
            "audio_tracks": sum(bool(s["voice"]) for s in plan["shots"]) + bool(plan["music"])
                            + len(plan["sound_stems"]),
            "captions_burned_in": plan["burn_subtitles"], "visual_profile": plan["visual_profile"],
            "automatic_music_ducking": mix_report.get("automatic_ducking", bool(
                plan["music"] and any(shot["voice"] for shot in plan["shots"]))),
            "review_status": "asset_approved_not_published" if master else "preview_unreviewed",
            "shot_reports": shot_reports, "audio_mastering": mix_report, "qc": qc,
            "required_clues": plan["required_clues"], "published": False,
        }
        proof = temp / "out.json"
        proof.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        committed = []
        try:
            for original, target in zip((srt, proof, staged), outputs):
                os.link(original, target)
                committed.append(target)
        except BaseException:
            for target in committed:
                target.unlink()
            raise
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description="Offline hybrid short-video studio, never posts to platforms")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--master", action="store_true", help="Require explicit per-asset approvals")
    p.add_argument("--cache", type=Path, help="Private resumable shot cache outside this repository")
    p.add_argument("--quality-check", action="store_true", help="Decode and inspect the actual final MP4")
    a = p.parse_args()
    print(json.dumps(produce(a.manifest, a.output, master=a.master, cache=a.cache,
                             quality_check=a.quality_check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
