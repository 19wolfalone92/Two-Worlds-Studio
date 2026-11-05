"""Real local media regressions. Coloured cards and tones are technical fixtures."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from studio.shorts_factory import Blocked, run, probe, sha, validate_project
from studio.shorts_review import (create_review, delivery_files, join_narration,
                                 render_html, write_review_archive)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ShortsReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.samples = []
        for index, seconds, color in ((1, .37, '#315478'), (2, .51, '#338a56')):
            image = Image.new('RGB', (400, 400), color)
            ImageDraw.Draw(image).rectangle((0, 0, 25, 400), fill='#ff3030')
            image.save(cls.root / f'{index}.png')
            count = round(seconds * 48000)
            values = (np.sin(np.arange(count) * .1) * 3500).astype('<i2')
            values[0], values[-1] = 9000, -9000
            stereo = np.stack([values, values], axis=1)
            cls.samples.append(stereo)
            with wave.open(str(cls.root / f'{index}.wav'), 'wb') as wav:
                wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(48000)
                wav.writeframes(stereo.tobytes())
        text = 'Тестовая реплика.'
        (cls.root / 'story.txt').write_text(text, encoding='utf-8')
        doc = {'schema': 1, 'id': 'review_fixture', 'title': 'Проверка сцен', 'language': 'ru',
               'script_file': 'story.txt', 'script_sha256': sha(cls.root / 'story.txt'),
               'width': 180, 'height': 320, 'fps': 30,
               'scenes': [{'id': n, 'image': f'{n}.png', 'voice': f'{n}.wav',
                           'source_text': text, 'camera': 'static',
                           'subtitles': [{'start': .01, 'end': .32, 'text': text}]} for n in (1, 2)]}
        cls.project = cls.root / 'project.json'
        cls.project.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
        cls.item = validate_project(cls.project, smoke=True)
        cls.result = run(cls.project, cls.root / 'exports', smoke=True)
        cls.folder = cls.root / 'exports/review_fixture'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.local = tempfile.TemporaryDirectory()
        self.addCleanup(self.local.cleanup)
        self.base = Path(self.local.name)

    def copy_delivery(self):
        dest = self.base / 'review_fixture'
        shutil.copytree(self.folder, dest)
        return dest

    def test_pcm_cache_has_no_aac_generation_or_limiter_leading_delay(self):
        clips = sorted((self.root / 'exports/.scene_cache/review_fixture').glob('*.mov'))
        self.assertEqual(len(clips), 2)
        info = probe(clips[0])
        audio = next(s for s in info['streams'] if s['codec_type'] == 'audio')
        self.assertEqual(audio['codec_name'], 'pcm_s16le')
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(clips[0]), '-map',
                                  '0:a:0', '-f', 's16le', '-'], capture_output=True, check=True)
        values = np.frombuffer(decoded.stdout, '<i2').reshape(-1, 2)
        self.assertEqual(len(values), 19200)
        self.assertGreater(abs(int(values[0, 0])), 1000, 'Lookahead delay lost the speech attack')
        self.assertGreater(abs(int(values[len(self.samples[0]) - 1, 0])), 1000,
                           'The final source sample was lost at the scene boundary')
        self.assertEqual(self.result['qa']['audio_mastering']['final_aac_generations'], 1)

    def test_narration_preserves_first_and_last_samples_and_only_adds_tail_padding(self):
        with wave.open(str(self.folder / 'narration_ru.wav'), 'rb') as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels()), (48000, 2))
            values = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').reshape(-1, 2)
        self.assertEqual(len(values), (12 + 16) * 1600)
        np.testing.assert_array_equal(values[:len(self.samples[0])], self.samples[0])
        self.assertFalse(values[len(self.samples[0]):19200].any())
        np.testing.assert_array_equal(values[19200:19200+len(self.samples[1])], self.samples[1])
        self.assertFalse(values[19200+len(self.samples[1]):].any())

    def test_mixed_master_keeps_the_next_scene_attack_on_its_exact_frame_boundary(self):
        with wave.open(str(self.folder / 'audio_master.wav'), 'rb') as wav:
            values = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').reshape(-1, 2)
        self.assertEqual(len(values), 28 * 1600)
        peak = int(np.argmax(np.abs(values[19200-500:19200+500, 0].astype('i4'))))
        self.assertEqual(peak, 500, 'An intermediate codec shifted speech away from the visual cut')

    def test_neural_take_edit_keeps_pcm_and_reports_actual_reuse(self):
        from studio.shorts_factory import render_scene
        clip = self.base / 'take.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                        'testsrc2=size=180x320:rate=30:duration=1', '-an', '-c:v', 'libx264',
                        '-threads', '1', '-pix_fmt', 'yuv420p', str(clip)], check=True)
        scene = dict(self.item['entries'][0], motion_clips=[clip])
        target = self.base / 'scene.mov'
        render_scene(scene, target, 180, 320, 30)
        self.assertEqual(next(s for s in probe(target)['streams'] if s['codec_type']=='audio')['codec_name'], 'pcm_s16le')
        receipt = json.loads(target.with_suffix('.edit.json').read_text())
        self.assertFalse(receipt['takes_reused'])

    def test_timeline_uses_encoded_integer_boundaries_including_first_frame_after_cut(self):
        data = json.loads((self.folder / 'timeline.json').read_text())
        self.assertEqual(data['total_frames'], 28)
        self.assertEqual(data['scenes'][0]['sample_frames'], [0, 6, 11])
        self.assertEqual(data['scenes'][1]['sample_frames'], [12, 20, 27])
        self.assertEqual(data['scenes'][1]['start_s'], 12 / 30)
        self.assertAlmostEqual(data['scenes'][1]['cues'][0]['start_s'], .41)
        with Image.open(self.folder / data['scenes'][1]['board']) as board:
            # Second tile is the first actual MP4 frame of scene 2, not scene 1.
            r, g, b = board.getpixel((8 + 188 + 90, 120))
            self.assertGreater(g, b)
        self.assertTrue(self.result['qa']['av_delivery']['constant_frame_rate_verified'])

    def test_review_reference_contains_the_whole_uncropped_source(self):
        data = json.loads((self.folder / 'timeline.json').read_text())
        with Image.open(self.folder / data['scenes'][0]['board']) as board:
            r, g, b = board.getpixel((8 + 3, 160))
            self.assertGreater(r, g * 2)

    def test_detail_frame_preserves_the_encoded_resolution(self):
        data = json.loads((self.folder / 'timeline.json').read_text())
        for scene in data['scenes']:
            with Image.open(self.folder / scene['detail']) as image:
                self.assertEqual(image.size,(180,320))
        self.assertIn('review_assets/scene_002_detail.jpg',delivery_files(self.folder))

    def test_all_returned_paths_exist_after_atomic_export(self):
        self.assertTrue(all(Path(name).is_file() for name in self.result['files']))
        names = delivery_files(self.folder)
        self.assertIn('narration_ru.wav', names)
        self.assertIn('review_assets/scene_002.jpg', names)
        self.assertIn('audio_master.wav', names)

    def test_wrong_scene_clock_cannot_create_a_review_for_another_edit(self):
        wrong = dict(self.item)
        wrong['entries'] = [dict(s) for s in self.item['entries']]
        wrong['entries'][1]['seconds'] += .1
        with self.assertRaisesRegex(RuntimeError, 'frame count'):
            create_review(wrong, self.folder)

    def test_exterior_narration_export_never_overwrites_existing_wav(self):
        output = self.base / 'voice.wav'
        output.write_bytes(b'keep original')
        with self.assertRaises(FileExistsError):
            join_narration(self.item['entries'], output, 30)
        self.assertEqual(output.read_bytes(), b'keep original')

    def test_subtitle_cover_audio_and_qa_tampering_are_all_detected(self):
        dest = self.copy_delivery()
        for name in ('subtitles.srt', 'cover.jpg', 'narration_ru.wav', 'qa_report.json', 'review.html'):
            with self.subTest(name=name):
                original = (dest / name).read_bytes()
                (dest / name).write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum changed'):
                    delivery_files(dest)
                (dest / name).write_bytes(original)

    def test_missing_review_board_cannot_be_delivered_as_complete(self):
        dest = self.copy_delivery()
        (dest / 'review_assets/scene_001.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            delivery_files(dest)

    def test_removed_modern_hashes_cannot_fall_back_to_the_legacy_package(self):
        dest = self.copy_delivery()
        receipt = json.loads((dest / 'receipt.json').read_text())
        del receipt['delivery_sha256']
        (dest / 'receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'missing its file checksums'):
            delivery_files(dest)

    def test_receipt_cannot_export_an_unrelated_parent_file(self):
        dest = self.copy_delivery()
        receipt = json.loads((dest / 'receipt.json').read_text())
        receipt['delivery_sha256']['../unrelated.txt'] = '0' * 64
        (dest / 'receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            delivery_files(dest)

    def test_symlinked_board_is_not_a_complete_local_delivery(self):
        dest = self.copy_delivery()
        board = dest / 'review_assets/scene_001.jpg'
        outside = self.base / 'board.jpg'
        outside.write_bytes(board.read_bytes()); board.unlink(); board.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escaped'):
            delivery_files(dest)

    def test_archive_contains_all_self_contained_viewer_and_audio_files(self):
        output = self.base / 'review.zip'
        result = write_review_archive([self.folder], output)
        self.assertEqual(result['videos'], 1)
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            names = archive.namelist()
            self.assertIn('review_fixture/review.html', names)
            self.assertIn('review_fixture/narration_ru.wav', names)
            self.assertIn('review_fixture/review_assets/scene_002.jpg', names)
            self.assertEqual(archive.read('review_fixture/final.mp4'), (self.folder / 'final.mp4').read_bytes())
        self.assertFalse(result['published'])

    def test_archive_failure_does_not_leave_a_final_or_partial_zip(self):
        output = self.base / 'review.zip'
        with patch('zipfile.ZipFile.open', side_effect=OSError('Interrupted write')):
            with self.assertRaises(OSError):
                write_review_archive([self.folder], output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.base.iterdir()))

    def test_file_change_while_archiving_is_detected_and_not_published(self):
        dest = self.copy_delivery()
        output = self.base / 'review.zip'
        actual = zipfile.ZipFile.open
        def changed(archive, name, *args, **kwargs):
            if name.endswith('/final.mp4'):
                (dest / 'final.mp4').write_bytes(b'Changed during transfer')
            return actual(archive, name, *args, **kwargs)
        with patch('zipfile.ZipFile.open', changed):
            with self.assertRaisesRegex(ValueError, 'during archive'):
                write_review_archive([dest], output)
        self.assertFalse(output.exists())

    def test_change_between_validation_and_copy_uses_the_original_receipt_hash(self):
        dest = self.copy_delivery()
        output = self.base / 'review.zip'
        def changed(folder):
            names = delivery_files(folder)
            (dest / 'cover.jpg').write_bytes(b'Changed just after validation')
            return names
        with patch('studio.shorts_review.delivery_files', changed):
            with self.assertRaisesRegex(ValueError, 'during archive'):
                write_review_archive([dest], output)
        self.assertFalse(output.exists())

    def test_replaced_sources_or_reframing_cannot_reuse_a_finished_export(self):
        from studio.autonomous_batch import _review_package
        project = {'id':self.item['id']}
        good, _ = _review_package(self.folder,project,self.item['digest'],
                                 current_sources=self.item)
        self.assertTrue(good)
        private = self.base / 'inputs';private.mkdir()
        for name in ('project.json','story.txt','1.png','2.png','1.wav','2.wav'):
            shutil.copyfile(self.root/name,private/name)
        (private/'1.png').write_bytes((private/'2.png').read_bytes())
        changed = validate_project(private/'project.json',smoke=True)
        good, _ = _review_package(self.folder,project,self.item['digest'],current_sources=changed)
        self.assertFalse(good)
        shutil.copyfile(self.root/'1.png',private/'1.png')
        doc=json.loads((private/'project.json').read_text())
        doc['scenes'][0]['framing']={'source_focus':[.75,.4]}
        (private/'project.json').write_text(json.dumps(doc))
        changed = validate_project(private/'project.json',smoke=True)
        good, _ = _review_package(self.folder,project,self.item['digest'],
                                 current_sources=changed)
        self.assertFalse(good)

    def test_changed_input_during_render_cannot_commit_a_final_movie(self):
        private = self.base / 'inputs';private.mkdir()
        for name in ('project.json','story.txt','1.png','2.png','1.wav','2.wav'):
            shutil.copyfile(self.root/name,private/name)
        actual = create_review
        def changed(item,folder):
            result = actual(item,folder)
            (private/'1.png').write_bytes((private/'2.png').read_bytes())
            return result
        with patch('studio.shorts_review.create_review',changed):
            with self.assertRaisesRegex(Blocked,'changed during rendering'):
                run(private/'project.json',self.base/'exports',smoke=True)
        self.assertFalse((self.base/'exports/review_fixture').exists())

    def test_existing_archive_is_preserved(self):
        output = self.base / 'review.zip'; output.write_bytes(b'Previous archive')
        with self.assertRaises(FileExistsError):
            write_review_archive([self.folder], output)
        self.assertEqual(output.read_bytes(), b'Previous archive')

    def test_empty_batch_cannot_be_reported_as_a_finished_archive(self):
        output = self.base / 'review.zip'
        with self.assertRaises(ValueError):
            write_review_archive([], output)
        self.assertFalse(output.exists())


class ReviewTextTests(unittest.TestCase):
    def test_user_story_text_stays_inert_and_template_markers_are_preserved(self):
        text = '</script><img src=x onerror=alert(1)> __TITLE__ __DATA__'
        data = {'title': text, 'scenes': [{'cues': [{'text': text}]}]}
        result = render_html(data, 'Заголовок __DATA__')
        payload = re.search(r'<script id="reviewData" type="application/json">(.*?)</script>', result, re.S)[1]
        self.assertEqual(json.loads(payload), data)
        self.assertEqual(result.count('<script'), 2)
        self.assertIn('<title>Заголовок __DATA__ · STORY GLITCH</title>', result)


if __name__ == '__main__':
    unittest.main()
