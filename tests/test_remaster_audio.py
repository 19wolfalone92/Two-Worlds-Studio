import shutil
import subprocess
import tempfile,unittest,wave
from pathlib import Path
import numpy as np
from studio.audio_master import mix_timeline
from studio.delivery_qc import loudness
from studio.remaster_audio import SR,scene_wave,write_wave,wood_knocks

class RemasterAudioTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_short_dynamic_mix_meets_encoded_loudness_and_peak_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seconds = 11
            t = np.arange(seconds * SR) / SR
            envelope = np.maximum(0., np.sin(t * 11)) ** 1.5
            active = ((t > .6) & (t < 2.5)) | ((t > 4.) & (t < 5.5))
            voice = .27 * active * envelope * (.73 * np.sin(2 * np.pi * 183 * t) +
                                                 .27 * np.sin(2 * np.pi * 451 * t))
            source, master, aac = (root / name for name in ("take.wav", "master.wav", "master.m4a"))
            write_wave(source, .02 * np.sin(2 * np.pi * 180 * t) + voice)
            mix = mix_timeline([{"bus": "dialogue", "path": source, "start": 0,
                                 "duration": seconds, "gain": 1.}], seconds, master)
            self.assertIn("integrated_lufs", mix["mastered_pcm"])
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(master),
                            "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
            measured = loudness(aac)
            self.assertTrue(-18 <= measured["integrated_lufs"] <= -14, measured)
            self.assertLessEqual(measured["true_peak_dbtp"], -1.)
            with wave.open(str(master), "rb") as output:
                self.assertEqual(output.getnframes(), seconds * SR)

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
