"""CPU-only Kaggle T4×2 integration contract tests; no remote GPU or downloads."""
import hashlib
import json
import shutil
import struct
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

from PIL import Image

from studio.kaggle_shorts import build_kernel, verify_result
from studio.kaggle_shorts_worker import (
    KaggleBlocked, prepare_input, run_job, export_private_assets, _safe_extract)


class KaggleShortsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.queue = self.base / 'production_queue'
        self.story = self.queue / '13'
        self.story.mkdir(parents=True)
        text = 'Тестовая история.\n\n' + '\n\n'.join('Русский текст сцены.' for _ in range(8))
        (self.story / 'story.txt').write_text(text, encoding='utf-8')
        self.doc = {
            'schema':1, 'id':'work_13', 'title':'Тестовая история.', 'language':'ru',
            'script_file':'story.txt', 'script_sha256':hashlib.sha256((self.story/'story.txt').read_bytes()).hexdigest(),
            'width':1080, 'height':1920, 'fps':24,
            'scenes':[{'id':i,'image':f'images/scene_{i:02d}.png',
                       'voice':f'voices/scene_{i:02d}.wav','subtitles':[]}
                       for i in range(1,9)]}
        (self.story/'project.json').write_text(json.dumps(self.doc,ensure_ascii=False),encoding='utf-8')
        self.shots=[{'scene':i,'exact_source_text':'Русский текст сцены.',
                     'visual_prompt_ru':'Фотореалистичная сцена с человеком на фоне города и вечерних огней.',
                     'music_cue':'sad'} for i in range(1,9)]
        (self.story/'shot_plan.json').write_text(json.dumps(self.shots,ensure_ascii=False),encoding='utf-8')

    def _zip(self, path):
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for f in self.story.rglob('*'):
                if f.is_file():
                    z.write(f,'production_queue/13/'+str(f.relative_to(self.story)))

    def test_source_zip_preflight_preserves_locked_original(self):
        path=self.base/'source.zip';self._zip(path)
        work=prepare_input(path,self.base/'work','13')
        a=run_job(work,self.base/'out',self.base/'models',story='13',phase='preflight')
        self.assertEqual(a['status'],'PREFLIGHT_OK')
        self.assertEqual(a['scene_count'],8)
        self.assertEqual(a['inventory']['image'],8)
        self.assertEqual((work/'story.txt').read_bytes(),(self.story/'story.txt').read_bytes())
        self.assertTrue((self.base/'out/run_status.json').exists())

    def test_resume_images_and_voice_timings(self):
        path=self.base/'source.zip';self._zip(path)
        work=prepare_input(path,self.base/'work','13')
        def fake_image(prompt,dest,models,*,seed,width,height):
            dest.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(width,height),(90,100,130)).save(dest)
            return {'model':'OFFLINE_TEST_ONLY'}
        def fake_voice(text,dest,models,*,speaker,style,device):
            dest.parent.mkdir(parents=True,exist_ok=True)
            with wave.open(str(dest),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000)
                w.writeframes(struct.pack('<h',1000)*8000)
            return {'seconds':1.0,'subtitles':[{'start':0.05,'end':0.95,'text':text}]}
        r=run_job(work,self.base/'img',self.base/'models',story='13',phase='image',
                  max_scenes=2,image_backend=fake_image)
        self.assertEqual(len(r['made']),2)
        self.assertEqual(r['status'],'COMPLETE')
        self.assertEqual(r['inventory']['image'],6)
        self.assertEqual(run_job(work,self.base/'skip',self.base/'models',story='13',phase='image',
                      max_scenes=2,image_backend=fake_image)['skipped'],[1,2])
        b=run_job(work,self.base/'wav',self.base/'models',story='13',phase='voice',
                  max_scenes=2,voice_backend=fake_voice)
        self.assertEqual(len(b['made']),2)
        self.assertEqual(b['inventory']['voice'],6)
        prior=export_private_assets(work,self.base/'export.zip',b)
        self.assertGreater(prior['files'],5)
        new=prepare_input(path,self.base/'continued','13',recovery=self.base/'export.zip')
        info=json.loads((new/'project.json').read_text(encoding='utf-8'))
        self.assertEqual(info['scenes'][0]['subtitles'][0]['text'],'Русский текст сцены.')
        self.assertTrue((new/'images/scene_01.png').exists())
        self.assertTrue((new/'voices/scene_01.wav').exists())
        runreceipt=self.base/'check';runreceipt.mkdir()
        shutil.copy(self.base/'export.zip',runreceipt/'export.zip')
        (runreceipt/'two_worlds_kaggle_receipt.json').write_text(json.dumps({
            'reviewed':False,'published':False,'success':True,
            'asset_bundle':{'zip':'export.zip','sha256':hashlib.sha256((self.base/'export.zip').read_bytes()).hexdigest()}}),
            encoding='utf-8')
        self.assertEqual(verify_result(runreceipt,'13')['ready_to_publish'],False)

    def test_reject_stale_recovery_by_script_sha(self):
        path=self.base/'source.zip';self._zip(path)
        work=prepare_input(path,self.base/'work','13')
        export_private_assets(work,self.base/'prev.zip',{'status':'PREFLIGHT_OK'})
        (self.story/'story.txt').write_text('Изменённый сюжет!',encoding='utf-8')
        self.doc['script_sha256']=hashlib.sha256((self.story/'story.txt').read_bytes()).hexdigest()
        (self.story/'project.json').write_text(json.dumps(self.doc,ensure_ascii=False),encoding='utf-8')
        self._zip(self.base/'revised.zip')
        with self.assertRaisesRegex(KaggleBlocked,'another story|revision'):
            prepare_input(self.base/'revised.zip',self.base/'new','13',recovery=self.base/'prev.zip')

    def test_zip_slip_is_rejected(self):
        malicious=self.base/'bad.zip'
        with zipfile.ZipFile(malicious,'w') as z:
            z.writestr('../escape.txt','bad')
        with zipfile.ZipFile(malicious) as z, self.assertRaisesRegex(KaggleBlocked,'Unsafe'):
            _safe_extract(z,self.base/'bad-dir')

    def test_bounded_range_and_bad_gpu(self):
        with self.assertRaisesRegex(KaggleBlocked,'Bounded'):
            run_job(self.story,self.base/'out',self.base/'models',story='13',phase='preflight',max_scenes=0)
        with self.assertRaisesRegex(KaggleBlocked,'Choose GPU'):
            run_job(self.story,self.base/'out',self.base/'models',story='13',phase='preflight',gpu_id=8)

    def test_kernel_config_is_private_and_contains_no_scripts(self):
        source=Path(__file__).resolve().parent.parent
        package=build_kernel(source,self.base/'kernel',username='test-creator',dataset='test-creator/private-story-queue',
                             model_dataset='test-creator/private-models',phase='image',story='13')
        self.assertTrue(package['private'])
        self.assertFalse(package['story_script_embedded'])
        self.assertFalse(package['generation_started'])
        meta=json.loads((self.base/'kernel/kernel-metadata.json').read_text())
        self.assertEqual(meta['machine_shape'],'NvidiaTeslaT4')
        self.assertEqual(meta['enable_gpu'],'true')
        self.assertEqual(meta['is_private'],'true')
        self.assertEqual(len(meta['dataset_sources']),2)
        src=(self.base/'kernel/kernel_main.py').read_text()
        self.assertNotIn('Русский текст сцены.',src)
        self.assertNotIn('KAGGLE_API_TOKEN',src)

    def test_preflight_without_gpu_or_model_dataset(self):
        source=Path(__file__).resolve().parent.parent
        package=build_kernel(source,self.base/'kernel',username='test-creator',
                             dataset='test-creator/private-story-queue',phase='preflight',story='13')
        self.assertFalse(package['gpu_requested'])
        self.assertEqual(json.loads((self.base/'kernel/kernel-metadata.json').read_text())['enable_internet'],'false')

    def test_gpu_without_models_is_rejected_before_submission(self):
        source=Path(__file__).resolve().parent.parent
        with self.assertRaisesRegex(ValueError,'private mounted model dataset'):
            build_kernel(source,self.base/'kernel',username='test-creator',
                         dataset='test-creator/private-story-queue',phase='image',story='13')

    def test_recovery_rejects_missing_dataset(self):
        source=Path(__file__).resolve().parent.parent
        with self.assertRaisesRegex(ValueError,'Recovery ZIP requires'):
            build_kernel(source,self.base/'kernel',username='test-creator',
                         dataset='test-creator/private-story-queue',phase='preflight',story='13',
                         recovery_path='/kaggle/input/private-resume/export.zip')


if __name__=='__main__':unittest.main()