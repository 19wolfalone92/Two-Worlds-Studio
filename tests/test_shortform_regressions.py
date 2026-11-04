import tempfile, unittest
from pathlib import Path
from studio.autonomous_factory import mood_of
from tools.kaggle_voice_batch_entry import find_recovery

class ShortformRegressions(unittest.TestCase):
    def test_explicit_genre_styles_are_not_silently_neutralized(self):
        for mood in ['neutral','tender','sad','mysterious','tense','hopeful']:
            with self.subTest(mood=mood):
                self.assertEqual(mood_of({'music_cue':mood}),mood)

    def test_cached_model_story_is_not_reused_for_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stale=root/'model-cache'/'story_14'/'TwoWorlds_PRIVATE_14_voice.zip'
            current=root/'explicit-recovery'/'TwoWorlds_PRIVATE_14_voice.zip'
            for p in [stale,current]:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            self.assertEqual(find_recovery(root,14,'model-cache'),[current])
            self.assertEqual(find_recovery(root,13,'model-cache'),[])
