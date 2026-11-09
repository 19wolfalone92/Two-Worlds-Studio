import shutil,subprocess,tempfile,unittest
from pathlib import Path
from studio.shorts_factory import qa

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class SingleBlackFrameTest(unittest.TestCase):
    def test_one_black_frame_at_30fps_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'one_black.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=red:s=160x284:r=30:d=1',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1',
                '-vf',"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='eq(n,15)'",
                '-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(dest)],check=True)
            with self.assertRaisesRegex(RuntimeError,'black frames'):
                qa(dest,160,284,30)
