"""Private, executable ComfyUI neural video bridge for TWO WORLDS Studio.

Runs preinstalled *official* ComfyUI model workflows; no cloud API, no
automatic weight download, no publishing, no model-licence bypass.  By default
only a loopback ComfyUI server can be accessed (also via SSH tunnel).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from studio.release_gate import sha256
from studio.video_ai_plan import _reference

MODEL_FILES = {
    "wan22-5b": ("wan2.2_ti2v_5B_fp16.safetensors",
                 "wan2.2_vae.safetensors",
                 "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    "ltx25-dfr": (),  # workflow validates installed official LTX components
}
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
            "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
            "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
MAX_DOWNLOAD = 512 * 1024 * 1024
ALLOWED_VIDEO = {".mp4"}  # Reject mislabeled WebM/MOV: importer requires an actual MP4


class NeuralBlocked(RuntimeError):
    """Stop rather than silently substituting a static/slideshow video."""


def _server(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("Use an explicit loopback HTTP ComfyUI origin; remote access needs an SSH tunnel")
    try:
        host = ipaddress.ip_address(parts.hostname or "")
        loopback = host.is_loopback
    except ValueError:
        loopback = (parts.hostname == "localhost")
    if not loopback or not parts.port or not 1 <= parts.port <= 65535:
        raise ValueError("ComfyUI must be local loopback (127.0.0.1, ::1 or localhost) with an explicit port")
    return url.rstrip("/")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise NeuralBlocked("HTTP redirects are disabled for private model asset transport")


class ComfyLocal:
    def __init__(self, origin: str, *, timeout: float = 60.0, opener=None):
        self.origin = _server(origin)
        self.opener = opener if opener is not None else build_opener(ProxyHandler({}), _NoRedirect())
        self.timeout = timeout

    def _read(self, path: str, *, data: bytes | None = None, content_type: str | None = None,
              maximum: int = 4 * 1024 * 1024) -> bytes:
        req = Request(self.origin + path, data=data,
                      headers={"Content-Type": content_type} if content_type else {},
                      method="POST" if data is not None else "GET")
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                header = response.headers.get("Content-Length")
                if header is not None and int(header) > maximum:
                    raise NeuralBlocked("ComfyUI response exceeds the bounded maximum")
                result = response.read(maximum + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise NeuralBlocked(f"Local ComfyUI request failed: {type(exc).__name__}") from exc
        if len(result) > maximum:
            raise NeuralBlocked("ComfyUI response was too large")
        return result

    def json(self, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        result = json.loads(self._read(path, data=body, content_type="application/json" if body else None))
        if not isinstance(result, dict):
            raise NeuralBlocked("Invalid ComfyUI JSON response")
        return result

    def upload_image(self, source: Path) -> str:
        # Direct multipart upload via the official /upload/image endpoint.
        boundary = "TW" + uuid.uuid4().hex
        raw = source.read_bytes()
        if len(raw) > 32 * 1024 * 1024:
            raise NeuralBlocked("Reference image exceeds 32 MiB")
        filename = "tw_" + sha256(source)[:32] + source.suffix.lower()
        payload = (
            ("--" + boundary + "\r\nContent-Disposition: form-data; name=\"image\"; filename=\"" +
             filename + "\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
            + raw +
            ("\r\n--" + boundary + "\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\nfalse"
             "\r\n--" + boundary + "--\r\n").encode()
        )
        response = self._read("/upload/image", data=payload,
                              content_type="multipart/form-data; boundary=" + boundary)
        data = json.loads(response)
        if not isinstance(data, dict) or not isinstance(data.get("name"), str) or (
                data.get("type", "input") != "input") or data.get("subfolder", "") not in ("", None):
            raise NeuralBlocked("ComfyUI did not return a plain uploaded input filename")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", data["name"]) or "/" in data["name"] or ".." in data["name"]:
            raise NeuralBlocked("ComfyUI returned an unsafe input filename")
        return data["name"]

    def submit(self, workflow: dict) -> str:
        result = self.json("/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())})
        ident = result.get("prompt_id")
        if result.get("error") or not isinstance(ident, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", ident):
            raise NeuralBlocked("ComfyUI rejected the video workflow; check model files and node inputs")
        return ident

    def result(self, prompt_id: str, *, max_wait: int = 3600, poll_s: float = 3.) -> dict:
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            job = self.json("/history/" + prompt_id).get(prompt_id)
            if isinstance(job, dict):
                status = job.get("status", {})
                if isinstance(status, dict) and status.get("status_str") in {"error", "interrupted"}:
                    raise NeuralBlocked("ComfyUI generation failed; review the private server logs")
                if isinstance(job.get("outputs"), dict) and job.get("outputs"):
                    if isinstance(status, dict) and status.get("completed") is False:
                        raise NeuralBlocked("ComfyUI emitted a partial video; generation was not completed")
                    return job
            time.sleep(poll_s)
        raise NeuralBlocked("ComfyUI generation exceeded its timeout (job may still run; inspect the server)")

    def download_video(self, job: dict, destination: Path) -> dict:
        candidates = []
        for node in job.get("outputs", {}).values():
            if not isinstance(node, dict):
                continue
            for name in ("videos", "gifs", "images"):
                for item in node.get(name, []):
                    if not isinstance(item, dict) or item.get("type") != "output":
                        continue
                    filename = item.get("filename", "")
                    subfolder = item.get("subfolder", "")
                    if (not isinstance(filename, str) or Path(filename).suffix.lower() not in ALLOWED_VIDEO
                            or not isinstance(subfolder, str) or ".." in filename or ".." in subfolder
                            or "/" in filename or "\\" in filename or subfolder.startswith(("/", "\\"))):
                        continue
                    candidates.append((filename, subfolder))
        if len(candidates) != 1:
            raise NeuralBlocked("Expected exactly one final video from an official SaveVideo node")
        filename, subfolder = candidates[0]
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        raw = self._read("/view?" + query, maximum=MAX_DOWNLOAD)
        if len(raw) < 4096:
            raise NeuralBlocked("ComfyUI video is empty or implausibly short")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".tw-neural-", suffix=Path(filename).suffix,
                                             dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(raw)
            return {"temporary": temporary, "origin_filename": filename, "download_bytes": len(raw)}
        except BaseException:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
            raise


def wan22_workflow(image_name: str, prompt: str, seed: int, *, width: int = 576,
                   height: int = 1024, frames: int = 121, steps: int = 50) -> dict:
    """Native ComfyUI API-format graph derived from Comfy-Org Wan2.2 5B template.

    Official UI template:
    https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json
    The dormant LoadImage branch is enabled as a real 9:16 I2V first frame.
    """
    if (not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", image_name) or ".." in image_name
            or not isinstance(prompt, str) or not 12 <= len(prompt.strip()) <= 1800
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(width) is not int or type(height) is not int
            or width < 256 or height < 448 or width % 16 or height % 16
            or abs(width / height - 9 / 16) > .02
            or type(frames) is not int or not 9 <= frames <= 241 or (frames - 1) % 4
            or type(steps) is not int or not 12 <= steps <= 50):
        raise ValueError("Invalid native Wan2.2-5B I2V parameters")
    return {
        "37": {"class_type": "UNETLoader", "inputs": {
            "unet_name": MODEL_FILES["wan22-5b"][0], "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": MODEL_FILES["wan22-5b"][2], "type": "wan", "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FILES["wan22-5b"][1]}},
        "56": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {
            "width": width, "height": height, "length": frames, "batch_size": 1,
            "vae": ["39", 0], "start_image": ["56", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": prompt.strip()}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": NEGATIVE}},
        "48": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["37", 0], "shift": 8.}},
        "3": {"class_type": "KSampler", "inputs": {
            "model": ["48", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["55", 0], "seed": seed, "steps": steps, "cfg": 5.,
            "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "57": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 24}},
        "58": {"class_type": "SaveVideo", "inputs": {
            "video": ["57", 0], "filename_prefix": "two_worlds/neural",
            "format": "auto", "codec": "auto"}},
    }


def template_workflow(path: Path, bindings: Path, *, image_name: str, prompt: str,
                      seed: int, width: int, height: int, frames: int) -> dict:
    """Use official LTX-2.5 DFR API-exported template with explicit slot bindings.

    Unlike ComfyUI UI exports, an API graph is keyed by node ID. Node IDs are
    not standardized; a user-exported mapping prevents silent wrong-node edits.
    """
    if path.stat().st_size > 2 * 1024 * 1024 or bindings.stat().st_size > 16 * 1024:
        raise ValueError("Template is too large")
    graph = json.loads(path.read_text(encoding="utf-8"))
    mapping = json.loads(bindings.read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not isinstance(mapping, dict) or not graph:
        raise ValueError("Expected API-format graph and explicit bindings")
    if not all(isinstance(v, dict) and isinstance(v.get("class_type"), str)
               and isinstance(v.get("inputs"), dict) for v in graph.values()):
        raise ValueError("Export File > Export Workflow (API), NOT the UI layout JSON")
    if not any(v["class_type"] == "SaveVideo" for v in graph.values()):
        raise ValueError("A real SaveVideo output node is required")
    changes = {"reference": image_name, "prompt": prompt, "seed": seed,
               "width": width, "height": height, "frames": frames}
    # Each binding is ["node_id", "input_name"]; no implicit edits, no eval.
    for name, value in changes.items():
        locator = mapping.get(name)
        if not (isinstance(locator, list) and len(locator) == 2
                and all(isinstance(x, str) for x in locator)
                and locator[0] in graph and locator[1] in graph[locator[0]]["inputs"]):
            raise ValueError(f"Missing valid API workflow binding for {name}")
        old = graph[locator[0]]["inputs"][locator[1]]
        if isinstance(old, list) or (name in {"seed", "width", "height", "frames"} and type(old) is not int):
            raise ValueError(f"Bound field {name} is not a scalar widget value")
        graph[locator[0]]["inputs"][locator[1]] = value
    # A generic graph with a SaveVideo node is NOT evidence of an LTX-2.5 DFR
    # model. Refuse to label a Wan/SDXL or plain distilled pass as flagship DFR.
    seen = json.dumps(graph, ensure_ascii=False)
    required = (
        "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
    )
    if any(component not in seen for component in required):
        raise ValueError("LTX-2.5 DFR workflow must reference the official distilled transformer, detailing LoRA and video VAE")
    return graph


def _probe(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        raise NeuralBlocked("Install ffprobe to verify generated neural MP4")
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_read_frames",
        "-of", "json", str(path),
    ], capture_output=True, text=True, timeout=120, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise NeuralBlocked("Generated file must contain one real video stream")
    s = streams[0]
    try:
        from fractions import Fraction
        fps = Fraction(s["avg_frame_rate"])
        n = int(s["nb_read_frames"])
        w, h = int(s["width"]), int(s["height"])
        seconds = float(s.get("duration", "nan"))
        if not math.isfinite(seconds):
            seconds = float(n / fps)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise NeuralBlocked("Cannot measure the decoded neural frame clock") from exc
    if n < 9 or not 8 <= fps <= 60 or w < 256 or h < 448 or abs(w / h - 9 / 16) > .035 or abs(seconds - float(n / fps)) > .12:
        raise NeuralBlocked("Neural take has invalid native video geometry or frame clock")
    return {"width": w, "height": h, "fps": str(fps), "frames": n, "seconds": round(seconds, 5)}


def import_take(clip: Path, receipt: Path, reference: Path, *, duration: float) -> dict:
    if receipt.stat().st_size > 64 * 1024:
        raise ValueError("Comfy neural receipt is oversized")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema") != 1
            or data.get("engine") != "neural_comfy"
            or data.get("model") not in MODEL_FILES
            or data.get("reference_sha256") != sha256(reference)
            or data.get("clip_sha256") != sha256(clip)
            or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("workflow_sha256")))
            or data.get("inference_performed") is not True
            or data.get("video") != _probe(clip)
            or data["video"]["seconds"] + .001 < duration):
        raise ValueError("Comfy take does not match the locked reference, provenance and full video")
    return {"model": data["model"], "clip_sha256": data["clip_sha256"],
            "native_video": data["video"], "character_review": "required",
            "weights_verified": False}


def generate(model: str, *, reference: Path, output: Path, prompt: str,
             seed: int, host: str, width: int, height: int,
             frames: int, steps: int, timeout: int, execute: bool,
             accepted_license: bool = False, template: Path | None = None,
             bindings: Path | None = None, client: ComfyLocal | None = None) -> dict:
    if model not in MODEL_FILES:
        raise ValueError("Unsupported reviewed model")
    reference, output = Path(reference).resolve(), Path(output).absolute()
    pic = _reference(reference)
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise ValueError("Choose a new, private, nonexisting MP4 filename")
    receipt = output.with_suffix(".receipt.json")
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError("Existing receipt would be overwritten")
    if not isinstance(prompt, str) or not 12 <= len(prompt.strip()) <= 1800:
        raise ValueError("Motion direction must contain 12–1800 characters")
    if (type(seed) is not int or not 0 <= seed < 2**32 or type(timeout) is not int
            or not 60 <= timeout <= 14400):
        raise ValueError("Invalid seed or finite GPU timeout")
    if not accepted_license:
        raise NeuralBlocked("Explicit model licence review is required (--accept-model-license)")
    if model == "wan22-5b":
        workflow = wan22_workflow("pending.png", prompt, seed, width=width,
                                  height=height, frames=frames, steps=steps)
    else:
        if template is None or bindings is None:
            raise ValueError("LTX-2.5 DFR requires a locally exported ComfyUI API workflow and binding map")
        if type(width) is not int or type(height) is not int or width % 16 or height % 16 or abs(width / height - 9 / 16) > .025:
            raise ValueError("Use a valid 9:16 resolution")
        if type(frames) is not int or not 9 <= frames <= 241:
            raise ValueError("Invalid frame count")
        workflow = template_workflow(template, bindings, image_name="pending.png", prompt=prompt,
                                     seed=seed, width=width, height=height, frames=frames)
    _server(host)
    plan = {"schema": 1, "engine": "neural_comfy", "model": model,
            "reference": pic, "reference_sha256": sha256(reference),
            "output": str(output), "seed": seed, "frames_requested": frames,
            "gpu_executed": False, "published": False}
    if not execute:
        return {"status": "PLANNED_NO_GPU", **plan}
    if client is None:
        client = ComfyLocal(host)
    image_name = client.upload_image(reference)
    if model == "wan22-5b":
        workflow["56"]["inputs"]["image"] = image_name
    else:
        workflow = template_workflow(template, bindings, image_name=image_name, prompt=prompt,
                                     seed=seed, width=width, height=height, frames=frames)
    graph_digest = hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest()
    prompt_id = client.submit(workflow)
    result = client.result(prompt_id, max_wait=timeout)
    downloaded = client.download_video(result, output)
    temporary = downloaded["temporary"]
    try:
        video = _probe(temporary)
        if model == "wan22-5b" and (
                video["frames"] != frames or video["fps"] != "24"
                or abs(video["width"] / video["height"] - width / height) > .03):
            raise NeuralBlocked("ComfyUI did not produce requested native Wan2.2 frame clock")
        # Atomic no-overwrite; if an unexpected process took the target name,
        # do not overwrite. The manifest is written next to exactly this clip.
        output.parent.mkdir(parents=True, exist_ok=True)
        os.link(temporary, output)
        proof = {"schema": 1, "engine": "neural_comfy", "model": model,
                 "model_file_names": MODEL_FILES[model],
                 "reference_sha256": plan["reference_sha256"],
                 "clip_sha256": sha256(output), "workflow_sha256": graph_digest,
                 "video": video, "seed": seed, "prompt_id": prompt_id,
                 "inference_performed": True, "weights_sha256": None,
                 "weights_verified": False, "human_approved": False, "published": False}
        try:
            with receipt.open("x", encoding="utf-8") as handle:
                json.dump(proof, handle, ensure_ascii=False, indent=2)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        return {"status": "GENERATED_REVIEW_REQUIRED", "output": str(output),
                "receipt": str(receipt), **proof}
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Execute locally installed open video models in ComfyUI")
    p.add_argument("--model", choices=tuple(MODEL_FILES), required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--host", default="http://127.0.0.1:8188")
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--frames", type=int, default=121)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--template", type=Path)
    p.add_argument("--bindings", type=Path)
    p.add_argument("--accept-model-license", action="store_true")
    p.add_argument("--execute-local-gpu", action="store_true")
    args = p.parse_args(argv)
    try:
        result = generate(args.model, reference=args.reference, output=args.output,
                          prompt=args.prompt, seed=args.seed, host=args.host,
                          width=args.width, height=args.height, frames=args.frames,
                          steps=args.steps, timeout=args.timeout,
                          template=args.template, bindings=args.bindings,
                          accepted_license=args.accept_model_license,
                          execute=args.execute_local_gpu)
    except (ValueError, NeuralBlocked, FileExistsError, OSError, subprocess.SubprocessError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc), "model": args.model,
                  "gpu_executed": False, "published": False}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
