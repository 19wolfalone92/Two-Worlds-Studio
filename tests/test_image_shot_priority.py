import unittest
from tools.kaggle_image_batch_entry import scene_prompt

class ShotPriorityTests(unittest.TestCase):
    def test_object_staging_precedes_human_portrait_language(self):
        director={'people':{},'scene_cast':{'1':[]},'palette':'cool gray and red','shot_first_prompt':True}
        prompt,cast=scene_prompt(director,{'scene':1,'visual_prompt_ru':'Only a red scarf and wet iron railing.'})
        self.assertEqual(cast,[])
        self.assertTrue(prompt.startswith('ONE vertical photorealistic film frame. EXACT SHOT: Only a red scarf'))
        self.assertIn('ZERO PEOPLE',prompt)
        self.assertNotIn('Preserve each referenced face',prompt)

    def test_hands_framing_is_not_buried_under_long_cast_descriptions(self):
        director={'people':{'a':{'name':'Father','description':'tired gray-haired man '*60}},'scene_cast':{'1':['a']},'palette':'warm','shot_first_prompt':True}
        prompt,cast=scene_prompt(director,{'scene':1,'visual_prompt_ru':'Hands and blank paper only; no face in view.'})
        self.assertLess(prompt.index('Hands and blank paper'),prompt.index('Reference image'))
        self.assertIn('never widen a hands-only',prompt)
        self.assertEqual(cast,['a'])

if __name__=='__main__':unittest.main()
