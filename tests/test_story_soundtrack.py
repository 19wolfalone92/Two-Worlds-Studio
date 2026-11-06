from pathlib import Path
import hashlib
import tempfile
import unittest
import wave
import numpy as np
from studio.story_soundtrack import normalize_voice,mix_tracks


class SoundtrackClockTests(unittest.TestCase):
    def test_loudness_processing_cannot_drop_the_last_video_frames(self):
        # Synthetic audio is only a regression fixture, never a production asset.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rate=24000;seconds=5.32
            rng=np.random.default_rng(7)
            samples=rng.normal(size=round(seconds*rate)).astype(np.float32)*.03
            def write(path,data):
                with wave.open(str(path),'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate)
                    wav.writeframes((data*32767).astype('<i2').tobytes())
            source=root/'source.wav';write(source,samples)
            original=hashlib.sha256(source.read_bytes()).hexdigest()
            frame_time=128/24
            normalized=root/'normalized.wav'
            normalize_voice(source,normalized,frame_time)
            bed=root/'bed.wav';write(bed,samples*.2)
            final=root/'mix.wav';mix_tracks(normalized,bed,bed,final,frame_time)
            for output in [normalized,final]:
                with wave.open(str(output),'rb') as wav:
                    self.assertEqual(wav.getnframes(),round(frame_time*48000))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),original)
