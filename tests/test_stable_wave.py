import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from studio.stable_wave import compose, foreground_envelope, render_wave, sha256
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2", "numpy"}:
        raise
    cv2 = None
    np = None


@unittest.skipUnless(cv2 is not None and np is not None,
                     "Optional OpenCV/NumPy are tested by the dedicated Studio media CI")
class StableWaveTests(unittest.TestCase):
    def test_unoccluded_background_fixed_but_person_remains_live(self):
        alpha=foreground_envelope(720,1280)
        self.assertEqual(alpha.shape,(1280,720))
        self.assertLess(float(alpha[40,50]),0.01)   # bookshelf, locked
        self.assertGreater(float(alpha[430,340]),0.99) # face and beard, live
        self.assertGreater(float(alpha[620,585]),0.99) # waving hand, live
        self.assertGreater(float(alpha[900,100]),0.99) # torso, live
        plate=np.zeros((1280,720,3),np.uint8)
        frame=np.full_like(plate,170)
        result=compose(frame,plate,alpha)
        self.assertEqual(int(result[40,50,0]),0)
        self.assertEqual(int(result[430,340,0]),170)
        self.assertEqual(int(result[620,585,0]),170)
        with self.assertRaises(ValueError):foreground_envelope(1080,1920)
        with self.assertRaises(ValueError):compose(frame,plate,alpha[:1])

    def test_original_must_never_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'source.mp4';fake.write_bytes(b'original-bytes')
            with self.assertRaises(ValueError):render_wave(fake,fake,Path(d)/'qa.json')
            self.assertEqual(fake.read_bytes(),b'original-bytes')
            with self.assertRaises(ValueError):render_wave(fake,Path(d)/'fresh.mp4',Path(d)/'qa.json',height=1000)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'real ffmpeg required')
    def test_real_video_full_decoding_motion_and_no_false_approval(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            src=root/'native_input.mp4'
            vid=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*'mp4v'),60,(720,1280))
            self.assertTrue(vid.isOpened())
            for i in range(150):
                img=np.zeros((1280,720,3),np.uint8)
                img[:200]=(30+i%6,80+i%3,140+i%3) # deliberately unstable shelf
                cv2.rectangle(img,(130,150),(650,1279),(110,170,195),-1)
                cv2.circle(img,(340,390),(125),(155,195,225),-1) # continuous head
                cv2.rectangle(img,(530+int(20*np.sin(i/12)),380),(600+int(20*np.sin(i/12)),830),(155,195,225),-1)
                vid.write(img)
            vid.release()
            original=sha256(src)
            dst=root/'out.mp4';report=root/'qa.json'
            result=render_wave(src,dst,report,height=1280)
            self.assertEqual(original,sha256(src))
            self.assertEqual(result['output_frames'],150)
            self.assertEqual(result['input_frames'],150)
            self.assertTrue(result['result_static_region']['decode_ok'])
            self.assertLess(result['result_static_region']['background_roi_mad_to_anchor_mean'],1.5)
            self.assertGreater(result['source_static_region']['background_roi_mad_to_anchor_mean'],
                               result['result_static_region']['background_roi_mad_to_anchor_mean'])
            self.assertFalse(result['human_artistic_approved'])
            self.assertFalse(result['native_neural_inference_performed'])
            self.assertEqual(result['synthetic_added_frames'],0)
            self.assertTrue(dst.exists())
            self.assertEqual(json.loads(report.read_text())['output_sha256'],sha256(dst))
            cap=cv2.VideoCapture(str(dst));captures=[]
            for i in (30,90):
                cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                ok,frame=cap.read();self.assertTrue(ok)
                captures.append(frame[350:850,475:665].copy())
            cap.release()
            self.assertGreater(np.mean(np.abs(captures[0].astype(np.int16)-captures[1].astype(np.int16))),0.5)

if __name__=='__main__':unittest.main()
