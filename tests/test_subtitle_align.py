import unittest
import numpy as np
from studio.subtitle_align import ctc_spans, readable_cues


class AlignmentTests(unittest.TestCase):
    def test_repeated_letters_require_real_blank_frames(self):
        p = np.full((7, 3), -15.0)
        for t, label in enumerate([0, 1, 1, 0, 1, 2, 0]):
            p[t, label] = -.01
        spans = ctc_spans(p, [1, 1, 2], 0)
        self.assertEqual(spans[0][:2], (1, 3))
        self.assertEqual(spans[1][:2], (4, 5))
        self.assertEqual(spans[2][:2], (5, 6))

    def test_too_little_audio_is_not_assigned_invented_timings(self):
        with self.assertRaises(ValueError):
            ctc_spans(np.zeros((1, 3)), [1, 1, 2], 0)

    def test_readable_groups_preserve_measured_intervals_and_words(self):
        text = 'Она вернулась домой. Он всё ещё ждал у окна.'
        words = [{'word': w, 'start': i*.5, 'end': i*.5+.4}
                 for i, w in enumerate(text.split())]
        cues = readable_cues(words, max_chars=27)
        self.assertEqual(' '.join(c['text'] for c in cues), text)
        self.assertEqual(cues[0]['start'], 0)
        self.assertEqual(cues[-1]['end'], words[-1]['end'])
        self.assertTrue(all(len(c['text']) <= 27 for c in cues))
