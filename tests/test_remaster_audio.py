import tempfile,unittest,wave
from pathlib import Path
import numpy as np
from studio.remaster_audio import SR,scene_wave,write_wave,wood_knocks

class RemasterAudioTests(unittest.TestCase):
    def test_exterior_silence_edit_preserves_native_speech_samples_and_clock(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'take.wav'
            source=np.zeros(3*SR,np.float32)
            t=np.arange(SR)/SR;source[SR:2*SR]=.2*np.sin(2*np.pi*143*t)
            write_wave(path,source)
            words=[{'word':'Привет.','start':1.0,'end':2.0}]
            edited,aligned,proof=scene_wave(path,words,first=True,last=True)
            self.assertEqual(proof['speech_speed_factor'],1.0)
            self.assertEqual(len(edited)%(SR//30),0)
            # Only samples outside speech may be removed; the actual take is exact.
            expected=(source[SR:2*SR]*32767).astype('<i2').astype(np.float32)/32768
            offset=round(aligned[0]['start']*SR)
            np.testing.assert_allclose(edited[offset:offset+SR],expected,atol=1e-6)
            self.assertGreaterEqual(len(edited)/SR-aligned[-1]['end'],.69)

    def test_missing_alignment_cannot_guess_cuts(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'take.wav';write_wave(p,np.ones(SR,np.float32)*.05)
            with self.assertRaisesRegex(ValueError,'alignment'):scene_wave(p,[])

    def test_three_knocks_have_three_distinct_onsets_and_no_clipping(self):
        sound=wood_knocks(1.0,[.03,.28,.53])
        self.assertEqual(sound.shape,(SR,2))
        self.assertLess(np.max(np.abs(sound)),1)
        self.assertEqual(float(np.max(np.abs(sound[:int(.03*SR)]))),0)
        for a,b in [(.03,.20),(.28,.45),(.53,.70)]:
            self.assertGreater(float(np.max(np.abs(sound[int(a*SR):int(b*SR)]))),.2)

if __name__=='__main__':unittest.main()
