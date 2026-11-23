import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from studio.eb01_anatomy_profile import (
    EXPERIMENT, PROFILE, anatomy_specs, constrain_reference, native_audio_mask_numpy, select_specs,
)
from tools.eb01_echo_parallel_pilot import prepare


class AnatomyProfileTests(unittest.TestCase):
    def test_paired_takes_isolate_anchors_and_fit_the_original_audio(self):
        for alias in ('owner', 'collaborator'):
            a, b = anatomy_specs(alias)
            for key in ('seed', 'prompt_sha256', 'context_key', 'segment_frames', 'portrait_sha256', 'voice_sha256'):
                self.assertEqual(a[key], b[key], key)
            self.assertEqual(a['frames'], sum(a['segment_frames']) - len(a['segment_frames']) + 1)
            self.assertGreaterEqual(a['frames'] / 25, a['voice_seconds'] + .2)
            self.assertEqual(hashlib.sha256(a['prompt'].encode()).hexdigest(), a['prompt_sha256'])
            self.assertFalse(a['anatomy_controls']['static_reference_boxes_xyxy'])
            self.assertTrue(b['anatomy_controls']['static_reference_boxes_xyxy'])

    def test_skull_mask_is_limited_to_jaw_tokens_in_all_latent_frames(self):
        spec = anatomy_specs('owner')[0]
        mask = native_audio_mask_numpy(spec, 33).reshape(9, 64, 36)
        expected = np.zeros_like(mask)
        expected[:, 12:15, 14:20] = 1
        np.testing.assert_array_equal(mask, expected)
        for frames in (-3, 0, 32):
            with self.assertRaises(ValueError):
                native_audio_mask_numpy(spec, frames)

    def test_original_reference_survives_a_deformed_continuation(self):
        spec = anatomy_specs('collaborator')[1]
        video = np.full((1, 3, 5, 1024, 576), .3, dtype='float32')
        mask = np.full((1, 1, 5, 1024, 576), 255, dtype='uint8')
        mask[:, :, 0] = 0
        canonical = np.ones_like(video[:, :, :1])
        constrain_reference(video, mask, canonical, spec)
        np.testing.assert_allclose(video[:, :, 0], .3)
        for x, y, X, Y in spec['anatomy_controls']['static_reference_boxes_xyxy']:
            self.assertTrue((video[:, :, 1:, y:Y, x:X] == 1).all())
            self.assertFalse(mask[:, :, 1:, y:Y, x:X].any())
        np.testing.assert_allclose(video[:, :, 1:, 200:230, 250:280], .3)
        self.assertTrue((mask[:, :, 1:, 200:230, 250:280] == 255).all())

    def test_anatomy_and_previous_pilots_keep_separate_kernel_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            ids = []
            for anatomy in (False, True):
                root = Path(temp) / str(anatomy)
                root.mkdir()
                folder, kernel = prepare(root, 'owner', 'exampleuser', '1234567', 'a'*40, {}, anatomy=anatomy)
                ids.append(kernel)
                self.assertEqual(json.loads((folder/'kernel-metadata.json').read_text())['is_private'], 'true')
            self.assertNotEqual(*ids)
        with self.assertRaises(ValueError):
            select_specs({'account_alias': 'owner', 'profile': PROFILE})
        self.assertEqual(len(select_specs({'account_alias': 'owner', 'profile': PROFILE, 'experiment': EXPERIMENT})), 2)
