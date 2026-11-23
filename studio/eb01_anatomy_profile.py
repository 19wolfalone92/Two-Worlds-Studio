"""Native regional controls for the two reviewed, locked-camera EB01 pilot shots."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from studio.eb01_echo_parallel_profile import PROFILE as BASE_PROFILE, specs as base_specs
from studio.eb01_echo_profile import replace_pinned

EXPERIMENT = 'anatomy_controls_v2'
PROFILE = 'echomimic-v3-native-paired-anatomy-v2'

# Coordinates in the actual 576x1024 native S01/S06 camera crops.
SHOTS = {
    'S06': {
        'speech': (224, 192, 320, 240),
        'anchors': {'upper_skull': (208, 54, 333, 174)},
        'segments': (33, 33, 25),
        'guidance': 1.25,
        'prompt': (
            'Locked camera. The seated skeleton answers. His lower jaw pivots gently '
            'at the hinges in time with his voice. The upper teeth, eye sockets and '
            'cheekbones move together as one solid skull. His shoulders settle.'
        ),
    },
    'S01': {
        'speech': (160, 128, 416, 384),
        'anchors': {'hand_screen_left': (75, 750, 219, 831),
                    'hand_screen_right': (371, 748, 515, 821)},
        'segments': (33, 33),
        'guidance': 2.0,
        'prompt': (
            'Locked camera. The interviewer asks a question. Her mouth moves with '
            'the words and her eyes shift toward her guest. Both hands rest apart '
            'on the desk, fingers relaxed. Her shoulders rise gently as she breathes.'
        ),
    },
}


def anatomy_specs(account):
    base = base_specs(account)[0]
    shot = SHOTS[base['shot']]
    prompt = shot['prompt']
    seed = 2026112300 if account == 'owner' else 2026112400
    specs = []
    for slot in (0, 1):
        controls = {
            'schema': 2,
            'audio_native_box_xyxy': list(shot['speech']),
            'static_reference_boxes_xyxy': [list(box) for box in shot['anchors'].values()] if slot else [],
            'audio_guidance_scale': shot['guidance'],
            'mode': 'SPATIAL_AUDIO_AND_CANONICAL_ANCHORS' if slot else 'SPATIAL_AUDIO_ONLY',
            'physical_rigidity_proven': False,
            'hand_anatomy_approved': False,
            'skull_anatomy_approved': False,
        }
        # Identical noise, prompt, speech and timing isolate the effect of anchors.
        specs.append(dict(base, profile=PROFILE, physical_gpu_index=slot, prompt=prompt,
                          prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          context_key=base['shot'] + '_ANATOMY', seed=seed,
                          segment_frames=list(shot['segments']),
                          frames=1 + sum(n - 1 for n in shot['segments']), anatomy_controls=controls))
    return specs


def select_specs(cfg):
    if cfg.get('experiment') is None and cfg['profile'] == BASE_PROFILE:
        return base_specs(cfg['account_alias'])
    if cfg.get('experiment') == EXPERIMENT and cfg['profile'] == PROFILE:
        return anatomy_specs(cfg['account_alias'])
    raise ValueError('Unknown EB01 native experiment/profile combination')


def native_audio_mask_numpy(spec, frames):
    """Area-weighted overlap on the actual 16-pixel transformer patch grid."""
    x0, y0, x1, y1 = spec['anatomy_controls']['audio_native_box_xyxy']
    if (spec['width'], spec['height']) != (576, 1024) or frames < 1 or frames % 4 != 1:
        raise ValueError('Unexpected native geometry or temporal compression')
    if not (0 <= x0 < x1 <= 576 and 0 <= y0 < y1 <= 1024):
        raise ValueError('Speech region outside the native crop')
    def coverage(size, start, end):
        axis = np.arange(0, size, 16)
        return (np.minimum(axis + 16, end) - np.maximum(axis, start)).clip(0, 16).astype(np.float32) / 16
    grid = coverage(1024, y0, y1)[:, None] * coverage(576, x0, x1)
    return np.broadcast_to(grid, (1 + (frames - 1) // 4, 64, 36)).copy().reshape(1, -1, 1, 1)


def mask_audio_residual(audio, mask):
    if mask is None:
        return audio
    if tuple(mask.shape) != (*audio.shape[:2], 1, 1):
        raise ValueError('Speech mask does not match the flattened video tokens')
    return audio * mask.to(device=audio.device, dtype=audio.dtype)


def constrain_reference(video, mask, canonical_frame, spec):
    """Keep the real continuation at t=0. Supply original pixels to the VAE at t>0."""
    if tuple(video.shape) != (1, 3, mask.shape[2], 1024, 576) or tuple(mask.shape) != (1, 1, video.shape[2], 1024, 576):
        raise ValueError('Unexpected native inpainting tensor dimensions')
    if tuple(canonical_frame.shape) != (1, 3, 1, 1024, 576):
        raise ValueError('Canonical reference has the wrong crop')
    for x0, y0, x1, y1 in spec['anatomy_controls']['static_reference_boxes_xyxy']:
        if not (0 <= x0 < x1 <= 576 and 0 <= y0 < y1 <= 1024):
            raise ValueError('Static reference outside the native crop')
        roi = (..., slice(y0, y1), slice(x0, x1))
        video[:, :, 1:][roi] = canonical_frame[roi]
        mask[:, :, 1:][roi] = 0


def patch_anatomy(runtime: Path):
    pipeline = runtime / 'src/pipeline_wan_fun_inpaint_audio_2512.py'
    source = pipeline.read_text()
    for name in ('audio_embeds', 'negative_audio_embeds'):
        context = f'context=(prompt_embeds, {name}, latent_model_input.shape[2], '
        source = replace_pinned(source, context + 'None)', context + 'ip_mask)')
    transform = runtime / 'src/wan_transformer3d_audio_2512.py'
    marker = '        audio_x = audio_x.view(b, latent_t, -1, n, d).view(b, -1, n, d)'
    transformed = replace_pinned(transform.read_text(), marker, marker + '\n'
                       '        from studio.eb01_anatomy_profile import mask_audio_residual\n'
                       '        audio_x = mask_audio_residual(audio_x, ip_mask)')
    pipeline.write_text(source)
    transform.write_text(transformed)
    return {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (pipeline, transform)}
