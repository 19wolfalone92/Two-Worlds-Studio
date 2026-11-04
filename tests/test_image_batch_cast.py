import unittest
from tools.kaggle_image_batch_entry import scene_prompt


class SceneCastingTests(unittest.TestCase):
    def test_off_scene_character_is_not_sent_as_identity_reference(self):
        director={'people':{'a':{'name':'ALICE','description':'auburn hair'},
                            'b':{'name':'BETH','description':'silver curls'}},
                  'scene_cast':{'1':['a']},'palette':'muted winter',
                  'scene_notes':{'1':'Exactly one person at the workbench.'}}
        prompt,cast=scene_prompt(director,{'scene':1,'visual_prompt_ru':'У верстака.'})
        self.assertEqual(cast,['a'])
        self.assertIn('ALICE',prompt)
        self.assertNotIn('BETH',prompt)
        self.assertNotIn('silver curls',prompt)
        self.assertIn('У верстака.',prompt)

    def test_missing_cast_identity_cannot_silently_change_the_character(self):
        with self.assertRaises(ValueError):
            scene_prompt({'people':{},'scene_cast':{'1':['unknown']}},
                         {'scene':1,'visual_prompt_ru':'scene'})
