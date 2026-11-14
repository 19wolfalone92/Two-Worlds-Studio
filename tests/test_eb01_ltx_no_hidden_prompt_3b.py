"""Regression against GPU stalls caused by silent heavy prompt-enhancer models.

LTX 2B is an image-to-video model; a short prompt must not silently turn
into Florence-2-large + Llama-3.2-3B inference on CPU under sequential offload.
"""
import inspect
import unittest
from pathlib import Path

import tools.kaggle_eb01_ltx2b_s02_fallback as pilot


class DistilledGPUActorPilotTests(unittest.TestCase):
    def test_official_large_captioner_and_llama_are_explicitly_disabled(self):
        text=inspect.getsource(pilot.main)
        self.assertIn('prompt_enhancement_words_threshold: 120',text)
        self.assertIn('prompt_enhancement_words_threshold: 0',text)
        self.assertIn('c.write_text(original_config.replace(',text)
        self.assertIn('original_config.count(threshold)!=1',text)

    def test_native_distilled_short_gpu_pilot_is_bounded_and_not_full_movie(self):
        text=inspect.getsource(pilot.main)
        self.assertIn('num_frames=33',text)
        self.assertIn('timeout=1400',text)
        self.assertIn('"mode":"pilot"',text)
        self.assertIn('"whole_movie_ready":False',text)
        self.assertIn('"human_artistic_approved":False',text)

    def test_frozen_source_and_full_movie_originals_stay_locked(self):
        text=inspect.getsource(pilot.main)
        self.assertIn('Lightricks/LTX-Video',text)
        self.assertIn('REVIEWED_SOURCE_SHA',text)
        self.assertIn('cfg.get("private") is True',text)
        self.assertIn('source_sha(reference)!=REVIEWED_SOURCE_SHA["S02"]',text)
        self.assertIn('exact_approved_prompt_unmodified',text)

if __name__=="__main__":
    unittest.main()
