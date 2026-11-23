"""Actual FFmpeg regression: native overlap join must preserve every pixel."""
import json,shutil,subprocess,tempfile,unittest
from pathlib import Path
from tools.eb01_echo_lossless_assembly import assemble,export_review

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class LosslessAssemblyTests(unittest.TestCase):
 def test_lossless_join_preserves_native_rgb_order_and_original_audio_packets(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);segments=[]
   for i in range(2):
    video=p/('fixture'+str(i)+'.mp4')
    # Public synthetic diagnostic fixtures, never passed off as model footage.
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
      'testsrc2=size=288x512:rate=25:duration=0.52','-vf','hue=h='+str(i*45),
      '-c:v','libx264','-crf','5','-pix_fmt','yuv420p',str(video)],check=True,capture_output=True,timeout=30)
    segments.append(video)
   audio=p/'fixture-audio.mp4'
   subprocess.run(['ffmpeg','-v','error','-i',str(segments[0]),'-f','lavfi','-i','sine=frequency=450:sample_rate=48000:duration=1',
     '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','1',str(audio)],check=True,capture_output=True,timeout=30)
   result=assemble(segments,audio,p/'assembled.mp4')
   self.assertEqual(result['probe']['frames'],25)
   self.assertTrue(all(result['checks'].values()))
   self.assertFalse(result['second_lossy_compression'])
   self.assertEqual(result['new_original_takes'],0)
   view=export_review(p/'assembled.mp4',p/'review.mp4')
   self.assertTrue(all(view['checks'].values()))
   self.assertFalse(view['review_is_lossless'])
   self.assertEqual(view['codec']['profile'],'High')

 def test_failed_verification_leaves_destination_available_for_retry(self):
  from unittest.mock import patch
  from tools.eb01_echo_lossless_assembly import join_video
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)
   segments=[]
   for i in range(2):
    video=root/f'segment{i}.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
      'testsrc2=size=288x512:rate=25:duration=0.52','-vf','hue=h='+str(i*45),
      '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True,timeout=30)
    segments.append(video)
   out=root/'joined.mp4'
   from tools import eb01_echo_lossless_assembly as assembly
   real_hashes=assembly.rgb_frame_hashes
   def changed_after_encoding(path):
    return ['a'*64] if path.name=='joined.mp4' else real_hashes(path)
   with patch.object(assembly,'rgb_frame_hashes',side_effect=changed_after_encoding):
    with self.assertRaisesRegex(ValueError,'changed decoded'):
     join_video(segments,out)
   self.assertFalse(out.exists())
   self.assertFalse(out.with_suffix('.receipt.json').exists())
   self.assertTrue(join_video(segments,out)['checks']['every_native_rgb_frame_identical'])
