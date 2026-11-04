"""Tests do not download model weights, use external APIs or require a GPU."""
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image

from studio.offline_media import (OfflineBlocked, split_narration, real_voice,
                                  real_image, tone_score, local_model)
from studio.autonomous_factory import inventory, run_story, atomic_json


class FakeVoiceModel:
    def generate_custom_voice(self, *, text, language, speaker, instruct):
        assert language == 'Russian'
        rate = 24000
        length = int(rate * .22)
        t = np.arange(length) / rate
        return [(.08 * np.sin(2 * np.pi * 240 * t)).astype('float32')], rate


class FakeVisualModel:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(images=[Image.new('RGB', (kwargs['width'],kwargs['height']), (74, 80, 90))])


class OfflineMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_phrase_splitting_does_not_change_words(self):
        source = 'Женщина посмотрела на часы. Уверена ли она, что всё правильно? Потом наступила тишина.'
        phrases = split_narration(source, 55)
        self.assertEqual(' '.join(phrases), source)
        self.assertTrue(all(len(x) <= 55 for x in phrases))
        with self.assertRaises(ValueError):
            split_narration('Only English text no Russian.', 55)

    def test_qwen_generated_wave_timestamps_measured(self):
        dst = self.base / 'voice.wav'
        result = real_voice('Я здесь. Но ты меня не слышишь.', dst, self.base,
                            model=FakeVoiceModel())
        self.assertTrue(dst.exists())
        with wave.open(str(dst)) as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertAlmostEqual(wav.getnframes()/wav.getframerate(), result['seconds'], delta=.005)
        self.assertEqual(' '.join(x['text'] for x in result['subtitles']),
                         'Я здесь. Но ты меня не слышишь.')
        self.assertEqual(result['alignment'], 'measured_synthesis_chunk_boundaries_not_word_alignment')
        with self.assertRaises(FileExistsError):
            real_voice('Я снова здесь.', dst, self.base, model=FakeVoiceModel())

    def test_audio_score_is_real_and_idempotent(self):
        p = self.base / 'score.wav'
        info = tone_score(p, 2.3, mood='mysterious', seed=11)
        self.assertTrue(p.is_file())
        self.assertEqual(info['source'], 'original_procedural_score')
        with wave.open(str(p)) as f:
            self.assertAlmostEqual(f.getnframes()/f.getframerate(), 2.3, delta=.002)
            self.assertEqual(f.getnchannels(), 1)
        with self.assertRaises(FileExistsError):
            tone_score(p, 2.3)

    def test_image_generation_no_fake_source_and_reference_hash(self):
        ref = self.base / 'ref.png'
        # With flat uniform ref: reference fingerprint is still audited.
        Image.new('RGB', (512,768), (100, 33, 12)).save(ref)
        fake = FakeVisualModel()
        p = self.base / 'image.png'
        with self.assertRaises(OfflineBlocked):
            real_image('detailed cinematic photograph of a grieving woman at night', p,
                       self.base, seed=42, pipe=fake,
                       references=[self.base/'missing.png'])
        # Fake visual is intentionally uniform -> quality gate must refuse.
        with self.assertRaisesRegex(OfflineBlocked, 'nearly blank'):
            real_image('detailed cinematic photograph of a grieving woman at night', p,
                       self.base, seed=42, pipe=fake, references=[ref])
        self.assertFalse(p.exists())
        self.assertEqual(fake.kwargs['image'].size, (512,768))

    def test_local_model_cannot_implicitly_download(self):
        with self.assertRaises(OfflineBlocked):
            local_model(self.base, 'image')
        (self.base/'image').mkdir()
        (self.base/'image/config.json').write_text('{}')
        self.assertEqual(local_model(self.base, 'image'), self.base/'image')

    def synthetic_project(self):
        folder = self.base / 'project'
        folder.mkdir()
        story = 'Тестовая история.'
        scenes = []
        shot_plan = []
        for i in range(1, 9):
            line = f'Это сцена номер {i}. Она рассказывает простую русскую историю.'
            shot_plan.append({'scene': i, 'exact_source_text': line,
                              'visual_prompt_ru': 'A cinematic photo of a woman in a warm room lit by a window, dramatic natural light.',
                              'music_cue': 'sad'})
            scenes.append({'id':i, 'image': f'images/scene_{i:02d}.png',
                           'voice': f'voices/scene_{i:02d}.wav', 'subtitles': []})
        story = '\n\n'.join([story] + [s['exact_source_text'] for s in shot_plan])
        (folder/'story.txt').write_text(story,encoding='utf-8')
        (folder/'project.json').write_text(json.dumps({
            'schema':1,'id':'demo99','title':'Тестовая история.',
            'script_file':'story.txt', 'script_sha256': __import__('hashlib').sha256(story.encode()).hexdigest(),
            'language':'ru','width':1080,'height':1920,'fps':24,'scenes':scenes},ensure_ascii=False),encoding='utf-8')
        (folder/'shot_plan.json').write_text(json.dumps(shot_plan,ensure_ascii=False),encoding='utf-8')
        return folder

    def test_offline_scene_resume_with_mock_real_files(self):
        root = self.synthetic_project()
        models = self.base/'models'
        for kind in ('image','voice'):
            (models/kind).mkdir(parents=True)
            (models/kind/'config.json').write_text('{}')
        from PIL import ImageDraw
        calls = {'images':0,'voices':0}
        def mock_image(prompt,path,model_dir,*,seed,references=None):
            calls['images'] += 1
            path.parent.mkdir(parents=True,exist_ok=True)
            im=Image.new('RGB',(768,1344),(90,110,140))
            dr=ImageDraw.Draw(im)
            dr.ellipse((200,100,600,800),fill=(198,155,111))
            im.save(path)
        def mock_voice(text,path,model_dir,*,speaker,style,device):
            calls['voices'] += 1
            return real_voice(text,path,model_dir,model=FakeVoiceModel(),speaker=speaker,style=style,device=device)
        with patch('studio.autonomous_factory.accelerator_status',return_value={'cuda':True,'gpu':'TEST','vram_gb':24}):
            first = run_story(root,self.base/'out',models,images=True,voices=True,
                              music=True,render=False,image_engine=mock_image,
                              voice_engine=mock_voice)
            second = run_story(root,self.base/'out',models,images=True,voices=True,
                               music=True,render=False,image_engine=mock_image,
                               voice_engine=mock_voice)
        self.assertEqual(first['created'],{'image':8,'voice':8,'music':8})
        self.assertEqual(second['created'],{'image':0,'voice':0,'music':0})
        self.assertEqual(calls,{'images':8,'voices':8})
        self.assertEqual(first['status'],'ASSETS_READY')
        self.assertTrue(inventory(root)['ready_for_video'])
        self.assertEqual(len(json.loads((root/'project.json').read_text())['scenes'][0]['subtitles']),1)
        self.assertFalse((self.base/'out').exists())

    def test_mismatched_locked_script_rejected_with_synthetic_fixture(self):
        root=self.synthetic_project()
        shots=json.loads((root/'shot_plan.json').read_text())
        shots[3]['exact_source_text']='Слова подменены.'
        atomic_json(root/'shot_plan.json',shots)
        with self.assertRaises(OfflineBlocked):
            inventory(root)

    def test_real_work_source_never_changed_by_empty_run(self):
        package = Path(__file__).resolve().parents[1] / 'demo_inputs/production_queue/13'
        if not package.exists():
            self.skipTest('Private source fixture intentionally absent from public GitHub')
        import shutil
        dest = self.base/'story'
        shutil.copytree(package, dest)
        before = (dest/'story.txt').read_bytes()
        status = inventory(dest)
        self.assertEqual(len(status['scenes']), 8)
        self.assertEqual(status['missing']['voice'], 8)
        result = run_story(dest, self.base/'out', self.base/'not_installed',
                           images=True, voices=True, music=True, render=True)
        self.assertEqual(result['status'],'ASSETS_PENDING')
        self.assertFalse((self.base/'out/work_13/final.mp4').exists())
        self.assertEqual((dest/'story.txt').read_bytes(), before)
        self.assertEqual(result['created']['image'],0)
        self.assertEqual(result['created']['voice'],0)
        self.assertEqual(result['missing']['image'],8)
        self.assertTrue((dest/'local_factory_receipt.json').exists())

    def test_malicious_script_or_shot_change_blocks(self):
        package = Path(__file__).resolve().parents[1] / 'demo_inputs/production_queue/13'
        if not package.exists():
            self.skipTest('Private source fixture intentionally absent from public GitHub')
        import shutil
        dest = self.base/'story'
        shutil.copytree(package, dest)
        shots = json.loads((dest/'shot_plan.json').read_text())
        shots[0]['exact_source_text'] = 'Поменяли сюжет!'
        atomic_json(dest/'shot_plan.json', shots)
        with self.assertRaisesRegex(OfflineBlocked,'Scene narration differs'):
            inventory(dest)


if __name__ == '__main__':
    unittest.main()