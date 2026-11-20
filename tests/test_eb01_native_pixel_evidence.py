"""Codec fixtures only: these tests never represent neural generation."""
import shutil,subprocess,tempfile,unittest
from pathlib import Path
from tools.eb01_native_pixel_evidence import decoded_pixel_evidence

@unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
class NativePixelEvidenceTests(unittest.TestCase):
    def evidence(self,pattern):
        with tempfile.TemporaryDirectory() as t:
            clip=Path(t)/'synthetic.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',pattern,
                '-frames:v','33','-c:v','libx264','-preset','ultrafast',str(clip)],
                capture_output=True,check=True,timeout=30)
            return decoded_pixel_evidence(clip,width=256,height=448)
    def test_frozen_frames_are_detected_despite_valid_33_frame_clock(self):
        report=self.evidence('color=black:size=256x448:rate=24')
        self.assertEqual(report['unique_decoded_rgb_frames'],1)
        self.assertEqual(report['adjacent_exact_duplicate_pairs'],32)
    def test_moving_fixture_has_33_distinct_decoded_frames(self):
        report=self.evidence('testsrc2=size=256x448:rate=24')
        self.assertEqual(report['unique_decoded_rgb_frames'],33)
        self.assertEqual(report['adjacent_exact_duplicate_pairs'],0)

if __name__=='__main__':unittest.main()
