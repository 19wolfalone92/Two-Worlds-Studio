import unittest
from studio.shorts_factory import frame_count

class FrameClockTests(unittest.TestCase):
    def test_ffprobe_microsecond_rounding_does_not_append_a_video_frame(self):
        for seconds,frames in [(53.466667,1604),(8.166667,245),(1.000001,30),(6.866667,206)]:
            with self.subTest(seconds=seconds):self.assertEqual(frame_count(seconds,30),frames)

    def test_actual_fractional_frame_is_still_padded(self):
        self.assertEqual(frame_count(1.034,30),32)
        self.assertEqual(frame_count(.1,30),3)

if __name__=='__main__':unittest.main()
