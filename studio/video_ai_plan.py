"""Offline, upstream-pinned quality plans for real neural video.

This module does not download weights, spend GPU quota, or pretend that
procedural motion has become neural inference. It only prepares commands
using flags checked against the upstream Wan2.2 generate.py entry point.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

# The exact checkpoint family matters: guidance, native FPS and steps differ.
# Source of truth:
# https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py
# https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_i2v_A14B.py
PROFILES = {
    "wan22_ti2v5b_quality": {
        "model": "Wan-AI/Wan2.2-TI2V-5B",
        "runtime": "Wan-Video/Wan2.2",
        "task": "ti2v-5B",
        "license": "Apache-2.0",
        "mode": "image_to_video",
        "native_fps": 24,
        "frames": 121,
        "duration_s": 121 / 24,
        "area": "1280*720",
        "steps": 50,
        "shift": 5.0,
        "guidance": 5.0,
        "reference_vram_gib": 24,
        "hardware_note": "Reference implementation 720p with offload targets an RTX 4090 24GB; no T4 16GB guarantee.",
        "upstream": "https://github.com/Wan-Video/Wan2.2",
        "status": "official_cli_plannable_gpu_unverified",
    },
    "wan22_i2va14b_quality": {
        "model": "Wan-AI/Wan2.2-I2V-A14B",
        "runtime": "Wan-Video/Wan2.2",
        "task": "i2v-A14B",
        "license": "Apache-2.0",
        "mode": "image_to_video",
        "native_fps": 16,
        "frames": 81,
        "duration_s": 81 / 16,
        "area": "1280*720",
        "steps": 40,
        "shift": 5.0,
        "guidance": "upstream_default_dual_expert_3.5_3.5",
        "reference_vram_gib": 80,
        "hardware_note": "High-memory reference 720p; quantized/community workers need separate verification.",
        "upstream": "https://github.com/Wan-Video/Wan2.2",
        "status": "official_cli_plannable_gpu_unverified",
    },
    "wan_animate2_distilled": {
        "model": "Wan-AI/Wan2.2-Animate-2-14B",
        "runtime": "Wan-Video/Wan-Animate-2",
        "license": "Apache-2.0",
        "mode": "reference_plus_driving_video",
        "native_fps": 24,
        "steps": 10,
        "guidance": 1.0,
        "flow_solver": "euler",
        "hardware_note": "Upstream tested 480p with 2x A800, 720p with 8x A800; not a Kaggle T4 preset.",
        "upstream": "https://github.com/Wan-Video/Wan-Animate-2",
        "status": "research_profile_no_unverified_runner",
    },
    "ltx25_dfr_quality": {
        "model": "Lightricks/LTX-2.5",
        "runtime": "Lightricks/LTX-2",
        "license": "LTX-2.x Community License (accept terms; check commercial eligibility)",
        "mode": "text_or_image_to_audio_video",
        "native_fps": 24,
        "frames": 121,
        "steps": "DFR_pipeline_managed",
        "width": 1024,
        "height": 1536,
        "hardware_note": "Upstream DFR uses distilled 22B + detailing IC-LoRA and spatial upscaler; ~66GiB staged components, GPU not preflighted.",
        "upstream": "https://github.com/Lightricks/LTX-2",
        "status": "research_profile_no_unverified_runner",
    },
    "musetalk15_facial_only": {
        "model": "TMElyralab/MuseTalk",
        "runtime": "TMElyralab/MuseTalk",
        "license": "MIT for code; verify dependency/model-weight terms",
        "mode": "audio_driven_face_inpainting_only",
        "face_crop_px": [256, 256],
        "hardware_note": "Lip synchronization alone cannot animate torso, jaw pose, hands or body contact.",
        "upstream": "https://github.com/TMElyralab/MuseTalk",
        "status": "research_profile_no_unverified_runner",
    },
}


def _reference(path: Path) -> dict:
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("I2V requires an existing PNG/JPEG/WebP reference image")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if (min(width, height) < 256 or max(width, height) > 8192
                or abs(width / height - 9 / 16) > .025):
            raise ValueError("Supply an already composed portrait 9:16 reference >= 256px")
        rgba = image.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] != 255:
            raise ValueError("Transparent cutouts need a finished opaque scene before I2V")
    return {"width": width, "height": height, "orientation": "9:16"}


def plan_wan_i2v(profile: str, *, runtime: Path, weights: Path, reference: Path,
                 output: Path, prompt: str, seed: int = 42,
                 free_vram_gib: float | None = None) -> dict:
    """Build an argv list, never execute it; no implicit fallback to a slideshow."""
    if profile not in {"wan22_ti2v5b_quality", "wan22_i2va14b_quality"}:
        raise ValueError("This upstream CLI runner is verified only for two Wan2.2 I2V tasks")
    spec = PROFILES[profile]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    if not isinstance(prompt, str) or not 12 <= len(prompt.strip()) <= 1800:
        raise ValueError("Provide a precise, shot-specific motion prompt (12-1800 characters)")
    if free_vram_gib is not None:
        if (type(free_vram_gib) not in (int, float) or not math.isfinite(free_vram_gib)
                or free_vram_gib < spec["reference_vram_gib"]):
            raise ValueError("Not enough measured free VRAM for the unmodified reference profile")
    runtime, weights, reference, output = (
        Path(runtime).resolve(), Path(weights).resolve(), Path(reference).resolve(),
        Path(output).absolute()
    )
    if not (runtime / "generate.py").is_file() or not weights.is_dir():
        raise ValueError("Stage the official upstream runtime and licensed local model weights")
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise ValueError("Use a new, nonexisting private .mp4 destination")
    picture = _reference(reference)
    argv = [
        "python", str(runtime / "generate.py"), "--task", spec["task"],
        "--ckpt_dir", str(weights), "--size", spec["area"],
        "--frame_num", str(spec["frames"]), "--sample_steps", str(spec["steps"]),
        "--sample_shift", str(spec["shift"]), "--sample_solver", "unipc",
        "--base_seed", str(seed), "--image", str(reference), "--prompt", prompt.strip(),
        "--save_file", str(output), "--offload_model", "True", "--t5_cpu",
    ]
    # The A14B MoE uses a pair of expert scales. Do not replace them with
    # one flat CFG parameter (which is appropriate for the 5B model only).
    if isinstance(spec["guidance"], float):
        argv.extend(["--sample_guide_scale", str(spec["guidance"])])
    return {
        "status": "READY_TO_RUN_BY_OPERATOR_NOT_EXECUTED",
        "profile": profile, "model": spec["model"], "license": spec["license"],
        "source_url": spec["upstream"], "reference": picture,
        "reference_path": str(reference), "output": str(output),
        "native_fps": spec["native_fps"], "frames": spec["frames"],
        "native_duration_s": spec["duration_s"],
        "picture_policy": "keep_native_motion; conform separately after QC",
        "body_motion_policy": "driving/body performance must be inspected; static mouth-only movement fails review",
        "estimated_hardware": spec["hardware_note"],
        "command_argv": argv,
        "weights_downloaded": False, "inference_performed": False,
        "artistic_approval": "required", "published": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline model quality presets and Wan I2V plan; never runs AI")
    parser.add_argument("--list", action="store_true", help="Print all reviewed profiles")
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--free-vram-gib", type=float)
    args = parser.parse_args(argv)
    if args.list:
        if args.profile or args.runtime or args.weights or args.reference or args.output:
            parser.error("--list cannot be combined with a model generation plan")
        result = {"profiles": PROFILES, "downloaded": False, "inference_performed": False}
    else:
        if any(value is None for value in
               (args.profile, args.runtime, args.weights, args.reference, args.output, args.prompt)):
            parser.error("--profile, --runtime, --weights, --reference, --output and --prompt are required")
        try:
            result = plan_wan_i2v(args.profile, runtime=args.runtime, weights=args.weights,
                                  reference=args.reference, output=args.output, prompt=args.prompt,
                                  seed=args.seed, free_vram_gib=args.free_vram_gib)
        except ValueError as exc:
            parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
