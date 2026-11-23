import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from studio.neural_motion import CHECKPOINT, REPO, _digest, import_take, keyframe_inputs


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class LTXKeyframeTests(unittest.TestCase):
    def test_endpoint_is_bound_at_import_and_one_frame_receipts_still_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, end, clip = root/'start.png', root/'end.png', root/'take.mp4'
            Image.new('RGB', (128, 160), 'green').save(first)
            Image.new('RGB', (128, 160), 'blue').save(end)
            self.assertEqual(keyframe_inputs(first, None, 64, 64, 9)['image'].size, (64, 64))
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=128x160:r=24',
                            '-frames:v', '9', '-c:v', 'libx264', str(clip)], check=True, timeout=30)
            proof = dict(model=REPO, checkpoint=CHECKPOINT, width=128, height=160, actual_neural_frames=9,
                         sha256=_digest(clip), source_image_sha256=_digest(first))
            record = root/'take.receipt.json'
            record.write_text(json.dumps(proof))
            self.assertEqual(import_take(clip, record, first, .35)['source_frames'], 9)
            with self.assertRaises(ValueError):
                import_take(clip, record, first, .35, end_reference=end)
            proof['end_image_sha256'] = _digest(end)
            record.write_text(json.dumps(proof))
            self.assertEqual(import_take(clip, record, first, .35, end_reference=end)['source_frames'], 9)
            with self.assertRaises(ValueError):
                import_take(clip, record, first, .35)
            with self.assertRaises(ValueError):
                import_take(clip, record, first, .35, end_reference=first)
