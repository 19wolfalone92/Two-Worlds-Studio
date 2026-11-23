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
            continue  # The entire scene is untrackable: report unknown, never good.
        new, forward, _ = cv2.calcOpticalFlowPyrLK(first, frame, points, None, **settings)
        if new is None or forward is None or new.shape != points.shape:
            continue  # Optical flow may return None after a feature disappears.
        old, current = points[:, 0], new[:, 0]
        forward_ok = (forward.reshape(-1) != 0) & np.isfinite(current).all(axis=1)
        if not forward_ok.any():
            continue
        # Never send failed/NaN forward tracks back into OpenCV: it can crash,
        # or make the inverse flow appear valid when there is no correspondence.
        idx = np.flatnonzero(forward_ok)
        safe_current = current[idx].reshape(-1, 1, 2).astype(np.float32)
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(
            frame, first, safe_current, None, **settings
        )
        if back is None or reverse is None or back.shape != safe_current.shape:
            continue
        inverse_ok = (reverse.reshape(-1) != 0)
        inverse_ok &= np.isfinite(back[:, 0]).all(axis=1)
        inverse_ok &= np.linalg.norm(back[:, 0] - old[idx], axis=1) <= 1.5
        valid = np.zeros(len(points), dtype=bool)
        valid[idx] = inverse_ok
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
    # A numerical screen is never proof of finger count, skull rigidity or acting.
    # Missing tracks and distorted regions both require review; no false green.
    return {'shot': shot, 'decoded_frames': count, 'regions': summary, 'review_flags': flags,
            'motion_screen_status': 'REVIEW_REQUIRED' if flags else 'METRIC_SCREEN_PASS_ONLY',
            'human_artistic_approved': False}


def compare_paired_motion(control: dict, anchored: dict) -> dict:
    """Compare matched A/B native pilots without mistaking tracking for anatomy proof.

    Only comparable region measurements (>=90% measured frames in BOTH takes)
    can establish numerical improvement. Missing tracks are not zero movement.
    """
    import math

    if (not isinstance(control, dict) or not isinstance(anchored, dict)
            or control.get('shot') != anchored.get('shot')
            or control.get('shot') not in SHOTS
            or control.get('decoded_frames') != anchored.get('decoded_frames')):
        raise ValueError('A/B must use the same verified shot and native frame clock')
    shot = control['shot']
    metrics = {}
    enough = True
    improved = True
    for region in SHOTS[shot]['anchors']:
        a = control.get('regions', {}).get(region, {})
        b = anchored.get('regions', {}).get(region, {})
        values = []
        for item in (a, b):
            if not isinstance(item, dict):
                values.append(None)
                continue
            frames = item.get('expected_comparisons')
            measured = item.get('measured_frames')
            p95 = item.get('p95_shape_residual_px')
            tracks = item.get('initial_tracks')
            if (type(frames) is not int or frames < 1 or
                    type(measured) is not int or measured < .9 * frames or measured > frames or
                    type(tracks) is not int or tracks < MIN_TRACKS or
                    type(p95) not in (int, float) or not math.isfinite(p95) or p95 < 0):
                values.append(None)
                continue
            values.append(float(p95))
        if None in values:
            enough = False
            improved = False
            metrics[region] = {'comparison': 'INSUFFICIENT_RELIABLE_TRACKING'}
            continue
        before, after = values
        gain = round(before - after, 3)
        meaningful = gain >= max(.25, .10 * before)
        improved &= meaningful
        metrics[region] = {'before_px': before, 'after_px': after, 'delta_px': gain,
                           'comparison': 'NUMERICAL_GAIN' if meaningful else 'NO_MEASURABLE_GAIN'}
    # An apparently rigid skull with a drifting background is not an improvement.
    backgrounds = []
    for take in (control, anchored):
        region = take.get('regions', {}).get('background', {})
        frames = region.get('expected_comparisons')
        n = region.get('measured_frames')
        p95 = region.get('p95_displacement_px')
        tracks = region.get('initial_tracks')
        if (type(frames) is int and frames >= 1
                and type(n) is int and n >= .9 * frames and n <= frames
                and type(tracks) is int and tracks >= MIN_TRACKS
                and type(p95) in (int, float) and math.isfinite(p95) and p95 >= 0):
            backgrounds.append(float(p95))
        else:
            backgrounds.append(None)
    if None in backgrounds:
        enough = False
        improved = False
        camera = {'comparison': 'INSUFFICIENT_RELIABLE_TRACKING'}
    else:
        before, after = backgrounds
        stable = after <= 3.0 and after <= before + .25
        improved &= stable
        camera = {'before_px': before, 'after_px': after,
                  'comparison': 'CAMERA_STABLE' if stable else 'CAMERA_DRIFT_OR_REGRESSION'}
    conclusion = ('INSUFFICIENT_EVIDENCE' if not enough else
                  'NUMERICAL_GAIN_NOT_ANATOMY_APPROVAL' if improved else 'NO_RELIABLE_IMPROVEMENT')
    return {'shot': shot, 'measured_regions': metrics, 'camera_stability': camera,
            'conclusion': conclusion, 'complete_tracking': enough,
            'numerically_improved': enough and improved,
            'human_artistic_approved': False, 'full_production_authorized': False}
