import shutil
import subprocess
import tempfile,unittest,wave
from pathlib import Path
from unittest.mock import patch
import numpy as np
from studio.audio_master import mix_timeline
from studio.delivery_qc import MAX_AAC_TRUE_PEAK_DBTP, dialogue_clock, loudness
from studio.hybrid_master import add_audio
from studio.remaster_audio import SR,scene_wave,write_wave,wood_knocks

class RemasterAudioTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_multiple_delayed_voices_keep_their_positions_in_the_encoded_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks, shots = [], []
            for index, (start, duration, freq) in enumerate(((.28, 2.56, 275), (3.83, 1.84, 590))):
                rate = 24000
                t = np.arange(round(duration * rate)) / rate
                edge = np.clip(np.minimum(t, duration - t) * 50, 0, 1)
                voice = .28 * edge * np.sin(2 * np.pi * (freq * t + 200 * t * t))
                path = root / f"voice_{index}.wav"
                write_wave(path, voice, rate)
                tracks.append({"bus": "dialogue", "path": path, "start": start,
                               "duration": duration, "gain": 1.})
                shots.append({"dialogue_text": "Locked take", "voice": path, "start": start,
                              "voice_offset": 0.})
            bed, effect = root / "bed.wav", root / "effect.wav"
            music = .035 * np.sin(2 * np.pi * 151 * np.arange(11 * SR) / SR)
            write_wave(bed, np.column_stack((music, music)))
            write_wave(effect, wood_knocks(.4, [.12]))
            tracks.extend([{"bus": "music", "path": bed, "start": 0, "duration": 11, "gain": 1.},
                           {"bus": "sfx", "path": effect, "start": 7.34, "duration": .4, "gain": .6},
                           {"bus": "sfx", "path": effect, "start": 9.21, "duration": .4, "gain": .4}])
            master, aac = root / "master.wav", root / "master.m4a"
            mix_timeline(tracks, 11, master)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(master),
                            "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
            timing = dialogue_clock(aac, shots, 30)
            self.assertEqual([r["status"] for r in timing], ["PASS", "PASS"], timing)
            self.assertTrue(all(abs(r["offset_ms"]) <= 1. for r in timing), timing)
            wrong = dialogue_clock(aac, [shots[0], {**shots[1], "start": 2.}], 30)
            self.assertEqual(wrong[1]["status"], "BLOCKED", wrong)

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

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_hybrid_master_reencodes_aac_overshoot_from_pcm_without_moving_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silent, output, voice = (root / name for name in ("silent.mp4", "final.mp4", "voice.wav"))
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
                            "-i", "color=c=blue:s=64x112:r=30:d=3",
                            "-an", "-c:v", "libx264", str(silent)], check=True)
            t = np.arange(SR) / SR
            envelope = np.sin(np.pi * t) ** 2
            write_wave(voice, .2 * envelope * np.sin(2 * np.pi * (173 * t + 60 * t * t)))
            shot = {"dialogue_text": "Locked take", "voice": voice, "start": 1.,
                    "voice_offset": .23, "voice_bus": "dialogue", "voice_duration": 1.,
                    "voice_cleanup": False}
            initial_aac = []

            def measure(path):
                result = loudness(path)
                if path == output:
                    initial_aac.append(result["integrated_lufs"])
                    # Exercise the retry on any FFmpeg version with real AAC output.
                    return {**result, "true_peak_dbtp": -.8}
                return result

            with patch("studio.delivery_qc.loudness", side_effect=measure):
                report = add_audio(silent, output, {"duration": 3., "upgraded": True,
                                   "sound_stems": [], "music": None, "shots": [shot]})
            self.assertEqual(report["aac_peak_correction_db"], .7)
            self.assertEqual(len(initial_aac), 1)
            corrected = loudness(output)
            self.assertAlmostEqual(initial_aac[0] - corrected["integrated_lufs"], .7, delta=.15)
            self.assertLessEqual(corrected["true_peak_dbtp"], MAX_AAC_TRUE_PEAK_DBTP)
            self.assertEqual(dialogue_clock(output, [shot], 30)[0]["status"], "PASS")

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
