"""Decoded optical flow relative to independent scene and facial anchors."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np


def _event(code, start, end, fps, component):
    return {"code": code, "severity": "REVIEW REQUIRED", "first_frame": start, "last_frame": end - 1,
            "start_seconds": round(start / fps, 4), "end_seconds": round(end / fps, 4),
            "component": component, "reproduce": "Inspect the decoded region at the indicated frame rate"}


def gray_frames(video, width, height, expected):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "2", "-i", str(video),
            "-map", "0:v:0", "-an", "-vf", f"scale={width}:{height}:flags=area,format=gray",
            "-fps_mode", "passthrough", "-frames:v", str(expected + 1), "-threads:v", "1",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as decoder:
        try:
            assert decoder.stdout is not None
            count = 0
            while raw := decoder.stdout.read(width * height):
                if len(raw) != width * height:
                    raise ValueError("Truncated frame in temporal QC")
                count += 1
                yield np.frombuffer(raw, np.uint8).reshape(height, width)
            if decoder.wait(timeout=15) or count != expected:
                raise ValueError(f"Temporal QC decoded {count} frames, expected {expected}")
        finally:
            if decoder.poll() is None:
                decoder.kill()
            decoder.wait()


def _track(cv, before, after, boxes):
    points, counts = [], []
    for x0, y0, x1, y1 in boxes:
        features = cv.goodFeaturesToTrack(before[y0:y1, x0:x1], maxCorners=55,
                                          qualityLevel=.02, minDistance=4, blockSize=5)
        count = 0 if features is None else len(features)
        if count:
            points.extend(features[:, 0, :] + [x0, y0])
        counts.append(count)
    if not points:
        empty = np.empty((0, 2), np.float32)
        return [(empty, empty) for _ in boxes]
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    options = {"winSize": (15, 15), "maxLevel": 2}
    target, forward, _ = cv.calcOpticalFlowPyrLK(before, after, source, None, **options)
    returned, reverse, _ = cv.calcOpticalFlowPyrLK(after, before, target, None, **options)
    valid = (forward.ravel() == 1) & (reverse.ravel() == 1)
    valid &= np.linalg.norm(source[:, 0, :] - returned[:, 0, :], axis=1) <= .42
    rows, start = [], 0
    for count in counts:
        end = start + count
        ix = slice(start, end)
        rows.append((np.ascontiguousarray(source[ix, 0][valid[ix]]),
                     np.ascontiguousarray(target[ix, 0][valid[ix]])))
        start = end
    return rows


def _fit(cv, pair):
    before, after = pair
    if len(before) < 6 or min(np.ptp(before, axis=0)) < 8:
        return None
    matrix, inliers = cv.estimateAffinePartial2D(
        before, after, method=cv.RANSAC, ransacReprojThreshold=.65, maxIters=350,
    )
    if matrix is None or int(inliers.sum()) < max(6, math.ceil(.55 * len(before))):
        return None
    return matrix


def _motion(pair, matrix):
    before, after = pair
    if matrix is None or len(before) < 4:
        return None
    residual = after - (before @ matrix[:, :2].T + matrix[:, 2])
    return float(np.percentile(np.linalg.norm(residual, axis=1), 75)), np.median(residual, axis=0)


def _stats(values):
    observed = [value for value in values if value is not None]
    if not observed:
        return {"pairs": 0}
    return {"pairs": len(observed), "median_px": round(float(np.median(observed)), 4),
            "p95_px": round(float(np.percentile(observed, 95)), 4), "max_px": round(float(max(observed)), 4)}


def inspect_flow(video: Path, regions: list[dict], width: int, height: int, fps: float,
                 expected: int, *, speech: dict | None = None, voice_offset: float = 0.,
                 assets: Path) -> dict:
    try:
        import cv2 as cv
    except ImportError:
        return {"status": "REVIEW REQUIRED", "events": [
            _event("optical_flow_unavailable", 0, expected, fps, "motion_qc")
        ], "regions": []}
    from studio.video_qc import _runs
    from studio.identity_qc import RegionAppearance

    factor = min(1., 432 / width, 768 / height)
    w, h = round(width * factor), round(height * factor)
    boxes = [tuple(round(v * d) for v, d in zip(r["box"], (w, h, w, h))) for r in regions]
    indices = {r["id"]: i for i, r in enumerate(regions)}
    camera = next((i for i, r in enumerate(regions) if r["kind"] == "camera"), None)
    anchors = [indices[r["relative_to"]] if r.get("relative_to") else camera for r in regions]
    fits = set(anchors) - {None}
    rows, appearances = [[] for _ in regions], {}
    for i, region in enumerate(regions):
        if "identity_reference" in region:
            x0, y0, x1, y1 = boxes[i]
            appearances[i] = (RegionAppearance(cv, assets, region["identity_reference"], (x1 - x0, y1 - y0)), [])
    speeds, previous = [], None
    stride = max(1, round(fps / 6))
    for index, frame in enumerate(gray_frames(video, w, h, expected)):
        if index % stride == 0 or index == expected - 1:
            for i, (appearance, samples) in appearances.items():
                x0, y0, x1, y1 = boxes[i]
                samples.append((index, appearance.compare(frame[y0:y1, x0:x1])))
        if previous is not None:
            tracks = _track(cv, previous, frame, boxes)
            matrices = {i: _fit(cv, tracks[i]) for i in fits}
            if camera is not None:
                matrix, (points, _) = matrices[camera], tracks[camera]
                speeds.append(None if matrix is None else float(np.median(np.linalg.norm(
                    points @ matrix[:, :2].T + matrix[:, 2] - points, axis=1))))
            for i, anchor in enumerate(anchors):
                if i != camera:
                    rows[i].append(_motion(tracks[i], matrices.get(anchor)))
        previous = frame

    reports = []
    for i, region in enumerate(regions):
        kind, events = region["kind"], []
        if kind == "camera":
            for start, end in _runs([v is None for v in speeds], max(3, math.ceil(.15 * fps))):
                events.append(_event("camera_registration_lost", start, end + 2, fps, "smooth_camera/cinematic"))
            for start, end in _runs([v is not None and v >= 4 for v in speeds], 1):
                events.append(_event("camera_jump_or_fast_move", start, end + 2, fps, "smooth_camera/cinematic"))
            reports.append({"id": region["id"], "camera_motion": _stats(speeds), "events": events})
            continue
        row = rows[i]
        values = [x[0] if x is not None else None for x in row]
        component = "facial_actor/lipsync" if kind in {"eyes", "mouth"} else "character_engine/deformable_actor"
        for window in region.get("motion_windows", []):
            start = math.ceil(window["start"] * fps - 1e-7)
            end = min(expected, math.ceil(window["end"] * fps - 1e-7))
            observed = values[start:end - 1]
            tracked = [v for v in observed if v is not None]
            if len(tracked) < max(2, .65 * len(observed)):
                events.append(_event("motion_evidence_unavailable", start, end, fps, component))
            elif sum(v >= .16 for v in tracked) < max(2, math.ceil(.10 * len(observed))):
                events.append(_event("no_independent_action", start, end, fps, component))
        for start, end in _runs([v is not None and v >= 4 for v in values], 1):
            events.append(_event("subject_jump_or_fast_action", start, end + 2, fps, component))
        reversals = []
        for a, b in zip(row, row[1:]):
            if a is None or b is None:
                reversals.append(False)
                continue
            x, y = np.linalg.norm(a[1]), np.linalg.norm(b[1])
            reversals.append(min(x, y) >= .28 and np.dot(a[1], b[1]) < -.5 * x * y)
        for start, end in _runs(reversals, 3):
            events.append(_event("repeated_direction_reversals", start, end + 3, fps, component))
        if kind == "mouth" and speech is not None:
            def quiet(index):
                time = (index + .5) / fps - voice_offset
                step = speech["step_ms"] / 1000
                return all(not 0 <= cell < len(speech["active"]) or not speech["active"][cell]
                           for cell in range(math.floor((time - .06) / step), math.ceil((time + .06) / step)))

            for start, end in _runs([v is not None and v >= .22 and quiet(j) for j, v in enumerate(values)],
                                    max(3, math.ceil(.12 * fps))):
                events.append(_event("mouth_motion_during_quiet_audio", start, end + 2, fps, component))
        report = {"id": region["id"], "relative_to": regions[anchors[i]]["id"] if anchors[i] is not None else None,
                  "residual_motion": _stats(values), "events": events}
        if i in appearances:
            _, samples = appearances[i]
            for start, end in _runs([v is None or v < 5 for _, v in samples], 2):
                events.append(_event("weak_reference_texture_match", samples[start][0],
                                     min(expected, samples[end][0] + stride), fps, "identity_qc"))
            report["identity_texture_matches"] = [
                {"seconds": round(frame / fps, 4), "geometric_inliers": value} for frame, value in samples
            ]
        reports.append(report)
    return {"status": "REVIEW REQUIRED" if any(r["events"] for r in reports) else "PASS",
            "method": "bidirectional_lucas_kanade_with_similarity_anchors",
            "analysis_dimensions": [w, h], "regions": reports, "events": []}
