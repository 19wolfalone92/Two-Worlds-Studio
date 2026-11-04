import json,math,shutil,subprocess,tempfile,unittest,wave
from pathlib import Path
from PIL import Image
from studio.cinematic_edit import shot_plan,render
from studio.shorts_factory import probe


class CinematicClockTests(unittest.TestCase):
    def test_short_takes_cannot_shorten_original_narration_clock(self):
        for seconds in [31.18,50.19,67.02]:
            plan=shot_plan(seconds,[81/24]*3,24)
            self.assertEqual(sum(s['frames'] for s in plan),math.ceil(seconds*24))
            self.assertTrue(any(s['kind']=='photo' for s in plan))
            self.assertEqual({s['clip'] for s in plan if s['kind']=='motion'},{0,1,2})

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg missing')
    def test_real_editor_output_preserves_complete_video_and_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'source.png';Image.new('RGB',(180,320),(48,95,122)).save(image)
            voice=root/'voice.wav'
            with wave.open(str(voice),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\x01\x00'*64000)
            clip=root/'take.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=180x320:rate=12',
                            '-t','0.5','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
            out=root/'edit.mp4'
            receipt=render({'seconds':4,'motion_clips':[clip]*3,'image':image,'voice':voice},out,180,320,12,None)
            info=probe(out);video=next(s for s in info['streams'] if s['codec_type']=='video')
            self.assertEqual(int(video['nb_frames']),48)
            self.assertAlmostEqual(float(info['format']['duration']),4,delta=.08)
            self.assertTrue(any(s['codec_type']=='audio' for s in info['streams']))
            self.assertEqual(receipt['unique_neural_input_seconds'],.5)
            self.assertTrue(receipt['photographs_are_not_character_animation'])


if __name__=='__main__':unittest.main()
