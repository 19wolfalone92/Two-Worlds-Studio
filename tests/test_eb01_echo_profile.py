"""No neural inference is simulated by these offline contract tests."""
import ast,tempfile,unittest
from pathlib import Path
from studio.eb01_echo_profile import PILOTS,pilot_spec,full_voice_frames,patch_upstream

class AudioDrivenProfileTests(unittest.TestCase):
    def test_full_original_voice_fits_each_true_native_clock(self):
        for alias in PILOTS:
            p=pilot_spec(alias)
            self.assertGreaterEqual(p['frames']/p['fps'],p['voice_seconds']+.2)
            self.assertEqual((p['frames']-1)%4,0)
            self.assertEqual((p['width'],p['height']),(576,1024))
            self.assertTrue(p['audio_is_model_input']);self.assertFalse(p['whole_movie_approved'])
    def test_longer_voice_is_extended_not_silently_cut(self):
        self.assertEqual(full_voice_frames(2.16),61)
        self.assertGreaterEqual(full_voice_frames(4.88)/25,5.08)
        with self.assertRaises(ValueError):full_voice_frames(float('nan'))
    def test_unknown_owner_does_not_get_another_accounts_work(self):
        with self.assertRaises(ValueError):pilot_spec('third')
    def test_unknown_upstream_does_not_accept_blind_text_patch(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'src').mkdir()
            (root/'src/wan_transformer3d_audio.py').write_text('def unexpected(): pass\n')
            with self.assertRaises(ValueError):patch_upstream(root)

if __name__=='__main__':unittest.main()
