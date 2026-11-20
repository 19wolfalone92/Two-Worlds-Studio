"""Bounded native T4 experiment; never approves production or actor performance."""
from __future__ import annotations
import hashlib
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from studio.eb01_v7_storyboard import SRC_SHA, SHOTS, sha

PROFILE = "native-576x1024-tiled-fp32-vae-v1"
BRANCH = "feature/eb01-kaggle-private-source-two-account-sync-20261116"


def native_reference(mount: Path, destination: Path, shot: str) -> dict:
    """Sample the original 704x1216 portrait directly, never a 320px derivative."""
    if shot not in ("S01", "S02"):
        raise ValueError("Only two bounded character pilots are authorized")
    scene = next(s for s in SHOTS if s[0] == shot)
    _, who, _, _, zoom, cx, cy, _ = scene
    candidates = [p for p in mount.rglob(who + ".png") if sha(p) == SRC_SHA[who]]
    if len(candidates) != 1 or destination.exists():
        raise ValueError("Missing/ambiguous original or existing native reference")
    with Image.open(candidates[0]) as original:
        if original.format != "PNG" or original.size != (704, 1216):
            raise ValueError("Canonical original geometry changed")
        image = original.convert("RGB")
        iw, ih = image.size
        rw, rh = int(iw / zoom), int(ih / zoom)
        if rw < 576 or rh < 1024:
            raise ValueError("Native source crop has insufficient original pixels")
        x = max(0, min(iw-rw, int((iw-rw)*cx*2)))
        y = max(0, min(ih-rh, int((ih-rh)*cy*2)))
        image = ImageOps.fit(image.crop((x, y, x+rw, y+rh)), (576, 1024),
                             method=Image.Resampling.LANCZOS)
        image = ImageEnhance.Color(image).enhance(.98)
        image = ImageEnhance.Contrast(image).enhance(1.03)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, optimize=True)
        destination.chmod(0o600)
    return {"original_sha256": SRC_SHA[who], "reference_sha256": sha(destination),
            "source_crop_size": [rw, rh], "native_size": [576, 1024],
            "low_resolution_reference_upscaled": False}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("Pinned LTX native-profile boundary changed")
    return text.replace(old, new)


def patch_native_sources(inference: Path, pipeline: Path, vae: Path) -> dict:
    """Build on the verified sequential FP16/FP32 policy, with exact source guards.

    Upstream tiled decode bypassed _decode(), losing latent normalization and
    timestep conditioning. Forward both through the same decoder as untiled.
    Do not enable temporal tiling: it loses causal context between chunks.
    """
    code = inference.read_text()
    marker = '    print("EB01_STAGE_END_MODEL_LOAD",flush=True)'
    code = replace_once(code, marker,
        '    pipeline.vae.set_tiling_params(sample_size=256, overlap_factor=0.25)\n'
        '    pipeline.vae.enable_hw_tiling()\n'
        '    pipeline.vae.disable_z_tiling()\n' + marker)
    inference.write_text(code)
    code = vae.read_text()
    code = replace_once(code,
        'def _hw_tiled_decode(self, z: torch.FloatTensor, target_shape):',
        'def _hw_tiled_decode(self, z: torch.FloatTensor, target_shape, timestep=None):')
    code = replace_once(code,
        '                tile = self.post_quant_conv(tile)\n'
        '                decoded = self.decoder(tile, target_shape=tile_target_shape)',
        '                decoded = self._decode(tile, target_shape=tile_target_shape, timestep=timestep)')
    code = replace_once(code,
        'self._hw_tiled_decode(z, target_shape)\n',
        'self._hw_tiled_decode(z, target_shape, timestep=timestep)\n')
    # The wrapper's original tile scale ignores the causal VAE's patchification.
    # Use actual model spatial_downscale_factor (32 for this checkpoint).
    code = replace_once(code,
        '        self.tile_latent_min_size = int(sample_size / (2 ** (num_blocks - 1)))',
        '        scale = getattr(self, "spatial_downscale_factor", 2 ** (num_blocks - 1))\n'
        '        self.tile_latent_min_size = int(sample_size / scale)')
    vae.write_text(code)
    code = pipeline.read_text()
    marker = '        # Remove the added conditioning latents'
    code = replace_once(code, marker,
        '        torch.cuda.empty_cache()\n' + marker)
    # Raw latent checkpoints stay inside the private notebook. A future resume
    # must verify its context; do not deserialize untrusted torch pickles.
    marker = '            image = vae_decode(\n'
    code = replace_once(code, marker,
        '            from safetensors.torch import save_file\n'
        '            save_file({"latents": latents.detach().cpu().contiguous(),\n'
        '                       "decode_timestep": (decode_timestep.detach().cpu().contiguous()\n'
        '                          if decode_timestep is not None else torch.zeros(1))},\n'
        '                      "/kaggle/working/EB01_NATIVE_PREDECODE.safetensors")\n' + marker)
    pipeline.write_text(code)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (inference, pipeline, vae)}
