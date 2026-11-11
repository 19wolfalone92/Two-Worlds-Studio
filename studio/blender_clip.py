"""Render a trusted local Blender scene as a silent portrait shot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

from PIL import Image

from studio.production_queue import private_path
from studio.release_gate import sha256
from studio.smooth_camera import encode_rgb_frames


def import_take(clip: Path, receipt: Path, duration: float) -> dict:
    """Verify a Blender worker's geometry and contact receipt at composition time."""
    if receipt.stat().st_size > 64 * 1024:
        raise ValueError("Blender take receipt is too large")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("status") != "rendered"
            or data.get("backend") != "blender_cycles_cpu" or data.get("clip_sha256") != sha256(clip)
            or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("scene_sha256")))
            or any(type(data.get(key)) is not int for key in ("frames", "fps", "width", "height", "first_frame"))
            or not 1 <= data["frames"] <= 300 or not 12 <= data["fps"] <= 30
            or data["frames"] / data["fps"] + .0001 < duration):
        raise ValueError("Blender receipt must match the exact full-length take")
    measured = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(clip)],
        capture_output=True, text=True, check=True, timeout=90,
    )
    try:
        stream, = json.loads(measured.stdout)["streams"]
        matches = (int(stream["nb_read_frames"]) == data["frames"]
                   and Fraction(stream["avg_frame_rate"]) == data["fps"]
                   and all(stream[k] == data[k] for k in ("width", "height")))
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError("Blender take has no usable native frame clock") from exc
    if measured.stderr or not matches:
        raise ValueError("Blender take differs from the worker frame clock")
    result = {key: data[key] for key in ("backend", "scene_sha256", "frames", "fps")}
    if "geometry_qc" in data:
        geometry = data["geometry_qc"]
        if not isinstance(geometry, dict) or geometry.get("frames_checked") != data["frames"]:
            raise ValueError("Blender geometry checks must cover the complete take")
        result["geometry_qc"] = geometry
    if "contact_qc" in data:
        contact = data["contact_qc"]
        if (not isinstance(contact, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(contact.get("manifest_sha256")))
                or not isinstance(contact.get("windows"), list) or not 1 <= len(contact["windows"]) <= 24):
            raise ValueError("Missing Blender contact evidence")
        for row in contact["windows"]:
            if (not isinstance(row, dict) or any(type(row.get(k)) is not int for k in
                                               ("start_frame", "end_frame", "frames_checked"))
                    or not data["first_frame"] <= row["start_frame"] < row["end_frame"] < data["first_frame"] + data["frames"]
                    or row["frames_checked"] != row["end_frame"] - row["start_frame"] + 1
                    or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                           for k in ("peak_gap_m", "peak_slip_m", "max_gap_m", "max_slip_m"))
                    or not 0 < row["max_gap_m"] <= .10 or not 0 < row["max_slip_m"] <= .10
                    or not 0 <= row["peak_gap_m"] <= row["max_gap_m"]
                    or not 0 <= row["peak_slip_m"] <= row["max_slip_m"]):
                raise ValueError("Blender contact evidence is incomplete or exceeds its tolerances")
        result["contact_qc"] = contact
    return result


def _object_name(value) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 120 and value.isprintable()


def _load_contacts(path: Path, first: int, last: int) -> tuple[list[dict], str]:
    if not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("Expected a readable contact manifest under 64 KiB")
    raw = path.read_bytes()
    doc = json.loads(raw)
    rows = doc.get("contacts") if isinstance(doc, dict) else None
    if (not isinstance(rows, list) or not 1 <= len(rows) <= 24
            or type(doc.get("schema")) is not int or doc["schema"] != 1):
        raise ValueError("Contact manifest requires schema 1 and 1–24 contact windows")
    names = set()
    keys = {"name", "marker", "target", "start_frame", "end_frame", "max_gap_m", "max_slip_m"}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != keys
                or any(not _object_name(row[k]) for k in ("name", "marker", "target"))
                or row["name"] in names or row["marker"] == row["target"]
                or any(type(row[k]) is not int for k in ("start_frame", "end_frame"))
                or not first <= row["start_frame"] < row["end_frame"] <= last
                or any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) or not 0 < row[k] <= .10
                       for k in ("max_gap_m", "max_slip_m"))):
            raise ValueError("Contact windows need unique names, distinct markers, rendered frames and meter tolerances")
        names.add(row["name"])
    return rows, hashlib.sha256(raw).hexdigest()


def _verify_external_assets(source: Path, manifest: Path) -> dict:
    if not manifest.is_file() or manifest.stat().st_size > 512 * 1024:
        raise ValueError("Expected a readable Blender asset manifest under 512 KiB")
    raw = manifest.read_bytes()
    document = json.loads(raw)
    assets = document.get("assets") if isinstance(document, dict) else None
    if (not isinstance(assets, dict) or not 1 <= len(assets) <= 1000
            or type(document.get("schema")) is not int or document["schema"] != 1):
        raise ValueError("Asset manifest requires schema 1 and 1–1000 asset hashes")
    for name, digest in assets.items():
        relative = Path(name)
        if (not 1 <= len(name) <= 250 or not name.isprintable() or "\\" in name
                or relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
                or not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None):
            raise ValueError("Asset manifest requires relative paths and lowercase SHA-256 values")
        path = (source.parent / relative).resolve()
        if not path.is_relative_to(source.parent) or not path.is_file():
            raise ValueError(f"Blender asset is missing or outside the project: {name}")
        if sha256(path) != digest:
            raise ValueError(f"Blender asset has a different SHA-256: {name}")
    return {"files": len(assets), "manifest_sha256": hashlib.sha256(raw).hexdigest()}


_CHECK_MESHES = """
from mathutils.bvhtree import BVHTree

def surface_tree(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            raise ValueError(f'Empty collision surface: {obj.name}')
        vertices = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        triangles = [tuple(tri.vertices) for tri in mesh.loop_triangles]
        return BVHTree.FromPolygons(vertices, triangles, all_triangles=True), triangles, vertices
    finally:
        evaluated.to_mesh_clear()

names = set(self_collision_meshes).union(name for pair in collision_pairs for name in pair)
objects = {name: scene.objects.get(name) for name in names}
if any(obj is None or obj.type != 'MESH' for obj in objects.values()):
    raise ValueError('Collision checks must name mesh objects in the active scene')
original_frame = scene.frame_current
for frame in checked_frames:
    scene.frame_set(frame)
    graph = bpy.context.evaluated_depsgraph_get()
    trees = {name: surface_tree(obj, graph) for name, obj in objects.items()}
    for left, right in collision_pairs:
        if trees[left][0].overlap(trees[right][0]):
            raise ValueError(f'TW_COLLISION: frame {frame}: {left} / {right}')
    for name in self_collision_meshes:
        tree, triangles, vertices = trees[name]
        other = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
        for left, right in tree.overlap(other):
            if left != right and set(triangles[left]).isdisjoint(triangles[right]):
                raise ValueError(f'TW_COLLISION: frame {frame}: {name} (self-intersection)')
scene.frame_set(original_frame)
"""

_CHECK_CONTACTS = """
import json
from pathlib import Path

unit_m = scene.unit_settings.scale_length
original_frame = scene.frame_current
markers = {name: scene.objects.get(name) for row in contacts for name in (row["marker"], row["target"])}
if any(obj is None or obj.type != 'EMPTY' for obj in markers.values()):
    raise ValueError('Contact checks require two Empty markers in the active scene')
reports = {}
origins = {}
for frame in range(min(row["start_frame"] for row in contacts), max(row["end_frame"] for row in contacts) + 1):
    active = [row for row in contacts if row["start_frame"] <= frame <= row["end_frame"]]
    if not active:
        continue
    scene.frame_set(frame)
    graph = bpy.context.evaluated_depsgraph_get()
    for row in active:
        name = row["name"]
        marker = markers[row["marker"]].evaluated_get(graph).matrix_world
        target = markers[row["target"]].evaluated_get(graph).matrix_world
        offset = target.to_quaternion().inverted() @ ((marker.translation - target.translation) * unit_m)
        origin = origins.setdefault(name, offset.copy())
        gap, slip = offset.length, (offset - origin).length
        if gap > row["max_gap_m"] or slip > row["max_slip_m"]:
            raise ValueError(f'TW_CONTACT: frame {frame}: {name}, gap {gap * 1000:.2f} mm, slip {slip * 1000:.2f} mm')
        report = reports.setdefault(name, {**row, "frames_checked": 0,
                                           "peak_gap_m": -1., "peak_slip_m": -1.})
        report["frames_checked"] += 1
        for label, value in (("gap", gap), ("slip", slip)):
            if value > report[f"peak_{label}_m"]:
                report[f"peak_{label}_m"] = value
                report[f"peak_{label}_frame"] = frame
scene.frame_set(original_frame)
Path(contact_report).write_text(json.dumps(list(reports.values())), encoding="utf-8")
"""

_SETUP_PASSES = """
import json
from pathlib import Path

layer = bpy.context.view_layer
layer.use_pass_combined = True
for flag in ('z', 'normal', 'ambient_occlusion', 'diffuse_direct', 'diffuse_indirect',
             'diffuse_color', 'glossy_direct', 'glossy_indirect', 'glossy_color'):
    setattr(layer, 'use_pass_' + flag, True)
layer.cycles.use_pass_volume_direct = layer.cycles.use_pass_volume_indirect = True
layer.cycles.denoising_store_passes = True
shadow = any(obj.is_shadow_catcher for obj in layer.objects if obj.type == 'MESH')
layer.cycles.use_pass_shadow_catcher = shadow
if scene.use_nodes and not scene.render.use_compositing:
    raise ValueError('Enable scene compositing to export render passes')
scene.use_nodes = True
scene.render.use_compositing = True
tree = scene.node_tree
for node in tree.nodes:
    if node.type == 'OUTPUT_FILE':
        node.mute = True
render_layer = tree.nodes.new('CompositorNodeRLayers')
render_layer.layer = layer.name
output_node = tree.nodes.new('CompositorNodeOutputFile')
output_node.base_path = passes_prefix
output_node.format.file_format = 'OPEN_EXR_MULTILAYER'
output_node.format.color_depth = '32'
output_node.format.exr_codec = 'ZIP'
output_node.layer_slots.clear()
channels = {
    'Beauty': 'Image', 'Depth': 'Depth', 'Normal': 'Normal', 'AO': 'AO',
    'DiffuseDirect': 'DiffDir', 'DiffuseIndirect': 'DiffInd', 'DiffuseColor': 'DiffCol',
    'GlossyDirect': 'GlossDir', 'GlossyIndirect': 'GlossInd', 'GlossyColor': 'GlossCol',
    'VolumeDirect': 'VolumeDir', 'VolumeIndirect': 'VolumeInd',
    'DenoiseAlbedo': 'Denoising Albedo', 'DenoiseNormal': 'Denoising Normal',
}
if shadow:
    channels['ShadowCatcher'] = 'Shadow Catcher'
for name, socket in channels.items():
    output_node.layer_slots.new(name)
    tree.links.new(render_layer.outputs[socket], output_node.inputs[name])
Path(passes_prefix + 'channels.json').write_text(json.dumps(list(channels)), encoding='utf-8')
"""


def render_clip(source: Path, output: Path, *, first_frame: int = 1, frames: int,
                width: int = 540, fps: int = 30, samples: int = 16,
                threads: int = 4, passes_output: Path | None = None,
                collision_pairs: tuple[tuple[str, str], ...] = (),
                self_collision_meshes: tuple[str, ...] = (),
                asset_manifest: Path | None = None, contact_manifest: Path | None = None) -> dict:
    """Use offline Cycles CPU, validate every frame, then atomically commit the MP4."""
    limits = ((first_frame, 1, 9700), (frames, 1, 300), (width, 180, 1080),
              (fps, 12, 30), (samples, 4, 128), (threads, 1, 16))
    if any(type(value) is not int or not lo <= value <= hi for value, lo, hi in limits):
        raise ValueError("Blender clip settings exceed the supported CPU render bounds")
    if width % 18:
        raise ValueError("Width must be a multiple of 18 for an even 9:16 canvas")
    if (not isinstance(collision_pairs, (list, tuple)) or len(collision_pairs) > 12
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                   or not all(_object_name(name) for name in pair)
                   or pair[0] == pair[1] for pair in collision_pairs)):
        raise ValueError("Specify up to 12 pairs of distinct collision mesh names")
    if (not isinstance(self_collision_meshes, (list, tuple)) or len(self_collision_meshes) > 8
            or not all(_object_name(name) for name in self_collision_meshes)):
        raise ValueError("Specify up to 8 self-collision mesh names")
    if source.suffix.lower() != ".blend" or not source.is_file():
        raise ValueError("A readable, trusted local .blend is required")
    if output.suffix.lower() != ".mp4" or output.is_symlink():
        raise ValueError("Choose a new private .mp4 output")
    source, output = source.resolve(), private_path(output)
    if output.exists():
        raise FileExistsError("Choose a new output path")
    asset_snapshot = _verify_external_assets(source, asset_manifest) if asset_manifest is not None else None
    if passes_output is not None:
        if passes_output.suffix.lower() != ".zip" or passes_output.is_symlink():
            raise ValueError("Render passes require a new private .zip output")
        passes_output = private_path(passes_output)
        if passes_output.parent != output.parent:
            raise ValueError("Keep the passes archive beside the output clip")
        if passes_output.exists():
            raise FileExistsError("Choose a new passes archive path")
    blender = shutil.which("blender")
    if not blender or not shutil.which("ffmpeg"):
        raise RuntimeError("Blender 4.3+ and FFmpeg must be available on PATH")

    height, end = width * 16 // 9, first_frame + frames - 1
    fingerprint = sha256(source)
    contacts, contact_sha = _load_contacts(contact_manifest, first_frame, end) if contact_manifest else ([], None)
    # Names and generated paths are inserted only as Python literals.
    settings = "\n".join((
        "import bpy",
        "scene = bpy.context.scene",
        "if scene.camera is None: raise ValueError('Select a scene camera first')",
        "r = scene.render",
        "r.engine = 'CYCLES'",
        "scene.cycles.device = 'CPU'",
        f"scene.cycles.samples = {samples}",
        f"r.resolution_x, r.resolution_y, r.resolution_percentage = {width}, {height}, 100",
        f"r.fps, r.fps_base = {fps}, 1.0",
        "r.pixel_aspect_x, r.pixel_aspect_y = 1, 1",
        "r.use_border = r.film_transparent = False",
        "r.use_sequencer = False",
        "r.image_settings.file_format = 'PNG'",
        "r.image_settings.color_mode = 'RGB'",
        "scene.frame_step = 1",
        f"r.threads_mode, r.threads = 'FIXED', {threads}",
    ))
    if collision_pairs or self_collision_meshes:
        settings += (f"\ncollision_pairs = {collision_pairs!r}\n"
                     f"self_collision_meshes = {self_collision_meshes!r}\n"
                     f"checked_frames = range({first_frame}, {end + 1})\n" + _CHECK_MESHES)

    output.parent.mkdir(parents=True, exist_ok=True)
    if passes_output is not None:
        # Reserve an uncompressed float32 upper estimate plus space for the ZIP copy.
        if shutil.disk_usage(output.parent).free < 2 * width * height * frames * 48 * 4 + 256 * 1024**2:
            raise RuntimeError("Insufficient free disk space for the EXR render passes")
    with tempfile.TemporaryDirectory(prefix=".tw-blender-", dir=output.parent) as directory:
        work = Path(directory)
        prefix = work / "shot_####"
        contact_report = work / "contacts.json"
        if contacts:
            settings += f"\ncontacts = {contacts!r}\ncontact_report = {str(contact_report)!r}\n" + _CHECK_CONTACTS
        if passes_output is not None:
            settings += f"\npasses_prefix = {str(work / 'layers_')!r}\n" + _SETUP_PASSES
        setup = work / "setup.py"
        setup.write_text(settings, encoding="utf-8")
        log = work / "blender.log"
        with log.open("wb") as diagnostics:
            process = subprocess.run([
                blender, "--background", "--factory-startup", "--offline-mode", "--disable-autoexec",
                str(source), "--python-exit-code", "10", "--python", str(setup),
                "-o", str(prefix), "-x", "1", "-s", str(first_frame), "-e", str(end), "-a",
            ], cwd=source.parent, stdout=diagnostics, stderr=subprocess.STDOUT, timeout=1800)
        if process.returncode:
            with log.open("rb") as diagnostics:
                diagnostics.seek(max(0, log.stat().st_size - 4096))
                tail = diagnostics.read().decode("utf-8", "replace")
            collision = next((line.partition("TW_COLLISION: ")[2] for line in tail.splitlines()
                              if "ValueError: TW_COLLISION: " in line), None)
            if collision:
                raise ValueError(f"Mesh collision at {collision}")
            contact = next((line.partition("TW_CONTACT: ")[2] for line in tail.splitlines()
                            if "ValueError: TW_CONTACT: " in line), None)
            if contact:
                raise ValueError(f"Contact failed at {contact}")
            reason = next((line for line in tail.splitlines()
                           if line.startswith(("ValueError: ", "RuntimeError: ", "Error: "))), "")
            raise RuntimeError(f"Blender could not prepare or render the scene (exit {process.returncode})"
                               + (f": {reason[:200]}" if reason else ""))
        filenames = [work / f"shot_{frame:04}.png" for frame in range(first_frame, end + 1)]
        if any(not path.is_file() for path in filenames):
            raise RuntimeError("Blender did not render the requested frame range. Check scene settings.")
        if sha256(source) != fingerprint:
            raise RuntimeError("The Blender scene changed during rendering")

        def images():
            for path in filenames:
                with Image.open(path) as frame:
                    yield frame.convert("RGB")

        temporary = work / "complete.mp4"
        encode_rgb_frames(temporary, width, height, fps, frames, images())
        digest = sha256(temporary)
        if asset_manifest is not None:
            if _verify_external_assets(source, asset_manifest) != asset_snapshot or sha256(source) != fingerprint:
                raise ValueError("Blender project inputs changed during the render")
        report = {"status": "rendered", "backend": "blender_cycles_cpu",
                  "first_frame": first_frame, "frames": frames, "fps": fps,
                  "width": width, "height": height, "scene_sha256": fingerprint,
                  "clip_sha256": digest, "published": False}
        if asset_snapshot is not None:
            report["external_assets"] = asset_snapshot
        if collision_pairs or self_collision_meshes:
            report["geometry_qc"] = {"frames_checked": frames, "mesh_pairs": len(collision_pairs),
                                     "self_collision_meshes": len(self_collision_meshes)}
        if contacts:
            if sha256(contact_manifest) != contact_sha:
                raise ValueError("The contact manifest changed during rendering")
            report["contact_qc"] = {"manifest_sha256": contact_sha,
                                    "windows": json.loads(contact_report.read_text(encoding="utf-8"))}
        if passes_output is None:
            os.link(temporary, output)
        else:
            exr_files = [work / f"layers_{frame:04}.exr" for frame in range(first_frame, end + 1)]
            channels = json.loads((work / "layers_channels.json").read_text(encoding="utf-8"))
            archive = work / "passes.zip"
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as container:
                for frame_path in exr_files:
                    with frame_path.open("rb") as stream:
                        if stream.read(4) != b"\x76\x2f\x31\x01":
                            raise ValueError("Invalid OpenEXR render pass frame")
                    container.write(frame_path, frame_path.name)
                container.writestr("manifest.json", json.dumps({**report, "channels": channels}))
            report["passes"] = {"frames": frames, "channels": channels, "sha256": sha256(archive)}
            os.link(archive, passes_output)
            try:
                os.link(temporary, output)
            except OSError:
                passes_output.unlink()
                raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--width", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--asset-manifest", type=Path,
                        help="JSON with SHA-256 checks for external project files")
    parser.add_argument("--contact-manifest", type=Path,
                        help="JSON contact windows between evaluated Blender Empty markers")
    parser.add_argument("--passes-output", type=Path, help="Multilayer float32 EXR frames in a ZIP")
    parser.add_argument("--collision-pair", dest="collision_pairs", nargs=2, action="append",
                        default=[], metavar=("MESH_A", "MESH_B"), help="Reject surface crossings")
    parser.add_argument("--self-collision-mesh", dest="self_collision_meshes", action="append",
                        default=[], metavar="MESH", help="Reject non-adjacent triangle crossings")
    try:
        result = render_clip(**vars(parser.parse_args()))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Blender clip blocked: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
