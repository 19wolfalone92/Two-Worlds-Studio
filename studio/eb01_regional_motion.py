"""Measure native S01/S06 motion against the first decoded frame. No artistic approval."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from studio.eb01_anatomy_profile import SHOTS
from studio.comfy_neural import _probe

# Static scenery inside the two reviewed crops. The speech ROI is free to move.
BACKGROUNDS = {'S01': (453, 30, 563, 178), 'S06': (15, 61, 172, 209)}
MIN_TRACKS = 6


def rigid_residual(origin, current):
    """Pixel residual after removing only an in-plane rotation and translation."""
    a = origin - origin.mean(axis=0)
    b = current - current.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return np.linalg.norm(a @ (u @ vt) - b, axis=1)


def _track_regions(first, frames, boxes):
    import cv2

    features, regions = [], {}
    for name, (x, y, X, Y) in boxes.items():
        window = first[y:Y, x:X]
        corners = cv2.goodFeaturesToTrack(window, maxCorners=64, qualityLevel=.012, minDistance=4)
        start = len(features)
        if corners is not None:
            features.extend(corners[:, 0] + [x, y])
        regions[name] = slice(start, len(features))
    points = np.array(features, dtype=np.float32).reshape(-1, 1, 2)
    history = {name: [] for name in boxes}
    count = 1
    settings = dict(winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01))
    for count, frame in enumerate(frames, start=2):
        if not len(points):
            continue
        new, forward, _ = cv2.calcOpticalFlowPyrLK(first, frame, points, None, **settings)
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(frame, first, new, None, **settings)
        old, current = points[:, 0], new[:, 0]
        valid = (forward[:, 0] != 0) & (reverse[:, 0] != 0)
        valid &= np.linalg.norm(back[:, 0] - old, axis=1) <= 1.5
        valid &= np.isfinite(current).all(axis=1)
        for name, section in regions.items():
            keep = valid[section]
            if keep.sum() < max(MIN_TRACKS, .60 * len(keep)):
                continue
            a, b = old[section][keep], current[section][keep]
            residual = rigid_residual(a, b)
            history[name].append((count - 1, float(np.quantile(residual, .90)),
                                  np.median(b - a, axis=0)))
    summary, flags = {}, []
    for name, rows in history.items():
        span = regions[name]
        summary[name] = item = {'initial_tracks': span.stop - span.start, 'measured_frames': len(rows),
                                'expected_comparisons': count - 1}
        if len(rows) < .9 * (count - 1) or not rows:
            flags.append(name + ':insufficient_tracking')
        if rows:
            worst = max(rows, key=lambda row: row[1])
            item.update(p95_shape_residual_px=round(float(np.quantile([r[1] for r in rows], .95)), 3),
                        worst_shape_frame=worst[0], worst_shape_residual_px=round(worst[1], 3),
                        p95_displacement_px=round(float(np.quantile([np.linalg.norm(r[2]) for r in rows], .95)), 3))
            if name not in ('speech', 'background') and item['p95_shape_residual_px'] > 2.5:
                flags.append(name + ':shape_change')
            if name == 'background' and item['p95_displacement_px'] > 3:
                flags.append('background:locked_camera_drift')
    return count, summary, flags


def inspect_regions(path: Path, shot: str) -> dict:
    import cv2

    if shot not in SHOTS:
        raise ValueError('Region coordinates require the reviewed S01 or S06 framing')
    meta = _probe(path)
    if (meta['width'], meta['height'], meta['fps']) != (576, 1024, '25'):
        raise ValueError('Region coordinates require native 576x1024 at 25 FPS')
    capture = cv2.VideoCapture(str(path))
    def grayscale():
        while True:
            ok, image = capture.read()
            if not ok:
                return
            yield cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        frames = grayscale()
        first = next(frames, None)
        if first is None:
            raise ValueError('Could not decode first native frame')
        boxes = dict(SHOTS[shot]['anchors'], background=BACKGROUNDS[shot], speech=SHOTS[shot]['speech'])
        count, summary, flags = _track_regions(first, frames, boxes)
    finally:
        capture.release()
    if count != meta['frames']:
        raise ValueError('Independent decoder did not read every native frame')
    return {'shot': shot, 'decoded_frames': count, 'regions': summary, 'review_flags': flags,
            'human_artistic_approved': False}
