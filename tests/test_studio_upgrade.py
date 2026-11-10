"""Regression cases from real short-form failures; fixtures are technical art."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from studio.caption_layout import CAPTION_WIDTH, ass_text, audit_cues, layout
from studio.cinematic_edit import shot_plan
from studio.delivery_qc import freeze_intervals, inspect_av, inspect_cover
from studio.shorts_factory import Blocked, probe, run, validate_project, write_ass, write_srt
from studio.smooth_camera import focal_transform, render_still, validate_framing
from studio.subtitle_align import readable_cues
from studio.voice_audit import audit_transcript, word_edits

FFMPEG = bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))


class CaptionLayoutTests(unittest.TestCase):
    def test_long_russian_caption_fits_two_lines_without_losing_words(self):
        text = 'Широкая строка субтитров проверяет переносы текста.'
        plan = layout(text)
        self.assertEqual(' '.join(plan['lines']), text)
        self.assertEqual(len(plan['lines']), 2)
        self.assertLessEqual(max(plan['widths']), CAPTION_WIDTH)
        self.assertGreaterEqual(plan['font_size'], 44)

    def test_impossible_caption_requires_regrouping_instead_of_truncation(self):
        with self.assertRaisesRegex(ValueError, 'regroup'):
            layout('Это важное слово. ' * 30)

    def test_literal_ass_controls_cannot_turn_source_text_into_drawing_commands(self):
        result = ass_text(r'Он написал {\p1} на листке.')
        self.assertNotIn(r'\p1', result)
        self.assertNotIn(r'{\p1}', result)

    def test_srt_numbering_and_source_words_survive_two_line_layout(self):
        text = 'Широкая строка субтитров проверяет переносы текста.'
        scene = {'seconds': 4., 'cues': [{'start': 0., 'end': 2., 'text': text},
                                          {'start': 2., 'end': 4., 'text': 'И всё же вернулись.'}]}
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'captions.srt'
            write_srt([scene], target, 30)
            blocks = target.read_text().strip().split('\n\n')
            self.assertEqual([b.splitlines()[0] for b in blocks], ['1', '2'])
            self.assertEqual(' '.join(blocks[0].splitlines()[2:]), text)

    def test_upper_caption_position_is_an_authored_option(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'captions.ass'
            write_ass([{'seconds': 2., 'caption_position': 'upper',
                        'cues': [{'start': 0., 'end': 1., 'text': 'Он вернулся.'}]}], target, 30)
            self.assertIn(r'\pos(500,380)', target.read_text())

    def test_measured_pause_is_not_covered_by_a_continuing_caption(self):
        words = [{'word': 'Он', 'start': 0., 'end': .2},
                 {'word': 'ждал.', 'start': .25, 'end': .6},
                 {'word': 'Долго.', 'start': 2., 'end': 2.6}]
        cues = readable_cues(words)
        self.assertEqual([(c['start'], c['end']) for c in cues], [(0., .6), (2., 2.6)])

    def test_long_spoken_phrase_is_split_on_real_word_boundaries(self):
        words = [{'word': 'слово', 'start': i * .7, 'end': i * .7 + .6} for i in range(12)]
        cues = readable_cues(words)
        self.assertTrue(all(c['end'] - c['start'] <= 3.2 for c in cues))
        self.assertEqual(' '.join(c['text'] for c in cues), ' '.join(w['word'] for w in words))
        boundaries = {w['end'] for w in words}
        self.assertTrue(all(c['end'] in boundaries for c in cues))

    def test_dense_speech_is_reported_without_speed_or_timing_changes(self):
        cues = [{'start': 1., 'end': 1.4, 'text': 'Важное признание героя.'}]
        report = audit_cues(cues)
        self.assertIn('fast_reading_load', report['warnings'][0]['issues'])
        self.assertFalse(report['speech_times_changed'])


class MeaningSensitiveVoiceTests(unittest.TestCase):
    def test_one_lost_negation_requires_review_even_when_total_wer_is_tiny(self):
        source = ' '.join(['Он долго ждал у двери.'] * 8) + ' Он не открыл её.'
        result = audit_transcript(source, source.replace('не открыл', 'открыл'))
        self.assertLess(result['word_error_rate'], .12)
        self.assertTrue(result['requires_review'])
        self.assertIn('negation_changed', [x['reason'] for x in result['critical_differences']])
        self.assertFalse(result['human_listening_performed'])

    def test_inserted_negation_and_changed_story_age_are_both_flagged(self):
        result = audit_transcript('Устройство работало двадцать минут.', 'Устройство не работало тридцать минут.')
        reasons = {x['reason'] for x in result['critical_differences']}
        self.assertTrue({'negation_changed', 'number_changed'} <= reasons)

    def test_authored_name_can_be_flagged_without_calling_asr_human_listening(self):
        result = audit_transcript('Сергей оставил ключ.', 'Андрей оставил ключ.', critical_terms=['Сергей'])
        self.assertTrue(any(x['reason'] == 'authored_term_changed' for x in result['critical_differences']))
        self.assertFalse(result['speech_quality_approved'])

    def test_capitalization_punctuation_and_yo_are_not_false_alarms(self):
        result = audit_transcript('Она всё ещё ждёт.', 'ОНА все еще ждет!')
        self.assertEqual(result['word_errors'], 0)
        self.assertFalse(result['requires_review'])

    def test_empty_recognition_is_missing_speech_not_a_success(self):
        result = audit_transcript('Он сохранил письмо.', '')
        self.assertEqual(result['word_error_rate'], 1.)
        self.assertTrue(result['requires_review'])

    def test_wrong_authored_term_cannot_silently_disable_name_checking(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            audit_transcript('Он сохранил письмо.', 'Он сохранил письмо.', critical_terms=['Питер'])

    def test_word_trace_counts_real_insertion_deletion_and_substitution(self):
        count, edits = word_edits(['a', 'b', 'c'], ['a', 'd', 'c', 'e'])
        self.assertEqual(count, 2)
        self.assertEqual([x['kind'] for x in edits], ['substitute', 'insert'])


class FocalCameraTests(unittest.TestCase):
    def test_off_center_path_stays_bounded_and_starts_and_stops_smoothly(self):
        config = {'source_focus': [.85, .4], 'screen_focus': [.46, .43]}
        path = [focal_transform(i, 90, 'pan_left', .45, (1920, 1080), (360, 640), config)
                for i in range(90)]
        for scale, _, x, _, _, y in path:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + 360 * scale, 1920 + 1e-8)
            self.assertLessEqual(y + 640 * scale, 1080 + 1e-8)
        movement = np.abs(np.diff([p[2] for p in path]))
        self.assertLess(movement[0], max(movement) * .04)
        self.assertLess(movement[-1], max(movement) * .04)

    def test_invalid_focus_is_rejected_before_any_render(self):
        for config in ({'source_focus': [float('nan'), .5]}, {'source_focus': [True, .5]},
                       {'source_focus': [.2, 2.]}, {'screen_focus': [.5, .5]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_framing(config)

    @unittest.skipUnless(FFMPEG, 'FFmpeg required')
    def test_real_focal_export_retains_a_prop_lost_by_center_crop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = Image.new('RGB', (960, 540), (35, 75, 110))
            ImageDraw.Draw(image).rectangle((800, 180, 900, 350), fill=(240, 25, 25))
            source = root / 'technical_prop.png'
            image.save(source)
            counts = []
            for name, framing in [('center', None), ('focus', {'source_focus': [.88, .49]})]:
                video = root / (name + '.mp4')
                render_still(source, video, 180, 320, 30, 15, 'static', framing=framing)
                raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video),
                                                '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
                rgb = np.frombuffer(raw, np.uint8).reshape(320, 180, 3)
                counts.append(int(((rgb[:, :, 0] > 160) & (rgb[:, :, 1] < 70)).sum()))
            self.assertEqual(counts[0], 0)
            self.assertGreater(counts[1], 1000)


class EditorialClockTests(unittest.TestCase):
    def test_two_available_takes_are_used_instead_of_freezing_take_zero(self):
        shots = shot_plan(8., [3., 6.], 30)
        self.assertEqual([(s['clip'], s['frames']) for s in shots], [(0, 90), (1, 150)])

    def test_neural_editor_uses_the_same_exact_frame_clock_as_the_still_renderer(self):
        shots = shot_plan(53.466667, [30., 30.], 30)
        self.assertEqual(sum(s['frames'] for s in shots), 1604)

    def test_reused_takes_do_not_receive_face_cropping_40_percent_zooms(self):
        shots = shot_plan(60., [3., 3., 3.], 30)
        self.assertLessEqual(max(s['zoom'] for s in shots), 1.04)


class DeliveryDiagnosticsTests(unittest.TestCase):
    def test_non_square_pixels_and_rotation_cannot_pass_as_portrait_delivery(self):
        for metadata in ({'sample_aspect_ratio':'4:3'}, {'side_data_list':[{'rotation':90}]}):
            info={'streams':[{'codec_type':'video',**metadata},
                             {'codec_type':'audio','channels':2,'sample_rate':'48000'}]}
            with self.subTest(metadata=metadata),self.assertRaisesRegex(RuntimeError,'portrait'):
                inspect_av(Path('unopened.mp4'),info,30)

    def test_intended_static_hold_is_not_described_as_a_broken_video(self):
        log = 'freeze_start: 2\nfreeze_end: 5\nfreeze_start: 8\n'
        report = freeze_intervals(log, 10., [(2., 3.), (2.5, 5.)])
        self.assertTrue(report[0]['authored_hold'])
        self.assertFalse(report[1]['authored_hold'])
        self.assertTrue(report[1]['requires_visual_review'])

    def test_a_jpg_file_is_not_evidence_of_a_checked_destination_cover(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'cover.jpg'
            Image.new('RGB', (540, 960), '#c34c36').save(path)
            result = inspect_cover(path)
            self.assertTrue(result['local_pixels_verified'])
            self.assertFalse(result['destination_cover_verified'])
            Image.new('RGB', (540, 960), 'black').save(path)
            with self.assertRaisesRegex(RuntimeError, 'near-black'):
                inspect_cover(path)

    @unittest.skipUnless(FFMPEG, 'FFmpeg required')
    def test_silent_encoded_aac_cannot_pass_on_codec_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'silent.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'color=c=red:s=160x284:r=30:d=1', '-f', 'lavfi', '-i',
                            'anullsrc=r=48000:cl=stereo', '-t', '1', '-c:v', 'libx264',
                            '-threads', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(target)], check=True)
            with self.assertRaisesRegex(RuntimeError, 'silent'):
                inspect_av(target, probe(target), 30)

    @unittest.skipUnless(FFMPEG, 'FFmpeg required')
    def test_variable_frame_clock_is_rejected_even_when_average_fps_is_30(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'vfr.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=s=160x284:r=30:d=2', '-f', 'lavfi', '-i',
                            'sine=frequency=440:sample_rate=48000:duration=2',
                            '-vf', "settb=1/90000,setpts='PTS+if(eq(N,20),900,0)'", '-fps_mode:v', 'vfr',
                            '-enc_time_base:v', '1:90000', '-c:v', 'libx264', '-threads', '1',
                            # Preserve the nominal packet duration with FFmpeg 7 and older muxers.
                            '-bf', '0', '-bsf:v', 'setts=duration=3000',
                            '-pix_fmt', 'yuv420p', '-ac', '2', '-c:a', 'aac', str(target)], check=True)
            info = probe(target)
            video = next(s for s in info['streams'] if s['codec_type'] == 'video')
            self.assertEqual(video['avg_frame_rate'], '30/1')
            with self.assertRaisesRegex(RuntimeError, 'constant-frame-rate'):
                inspect_av(target, info, 30)

    @unittest.skipUnless(FFMPEG, 'FFmpeg required')
    def test_audio_tail_one_second_longer_than_picture_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'drift.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'color=c=red:s=160x284:r=30:d=1', '-f', 'lavfi', '-i',
                            'sine=frequency=440:sample_rate=48000:duration=2', '-c:v', 'libx264',
                            '-threads', '1', '-pix_fmt', 'yuv420p', '-ac', '2', '-c:a', 'aac', str(target)], check=True)
            with self.assertRaisesRegex(RuntimeError, 'drift'):
                inspect_av(target, probe(target), 30)


@unittest.skipUnless(FFMPEG, 'FFmpeg required')
class SourceAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.voice = self.root / 'voice.wav'
        with wave.open(str(self.voice), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(48000)
            wav.writeframes((np.sin(np.arange(48000) * .1) * 3000).astype('<i2').tobytes())
        Image.new('RGB', (400, 720), (40, 80, 130)).save(self.root / 'frame.png')
        text = 'Технический тест.'
        (self.root / 'story.txt').write_text(text)
        self.doc = {'schema': 1, 'id': 'upgrade_fixture', 'title': text, 'language': 'ru',
                    'script_file': 'story.txt', 'script_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'width': 180, 'height': 320, 'fps': 30,
                    'scenes': [{'id': i, 'image': 'frame.png', 'voice': 'voice.wav',
                                'subtitles': [{'start': .05, 'end': .9, 'text': text}]} for i in (1, 2)]}
        self.project = self.root / 'project.json'
        self.save()

    def save(self):
        self.project.write_text(json.dumps(self.doc, ensure_ascii=False))

    def test_changed_focal_composition_invalidates_only_affected_scene_cache(self):
        run(self.project, self.root / 'out', smoke=True, cache_only=True)
        self.doc['scenes'][0]['framing'] = {'source_focus': [.75, .4]}
        self.save()
        from studio.shorts_factory import render_scene
        with patch('studio.shorts_factory.render_scene', wraps=render_scene) as renderer:
            run(self.project, self.root / 'out', smoke=True, cache_only=True)
        self.assertEqual([call.args[0]['id'] for call in renderer.call_args_list], [1])

    def test_changed_source_words_cannot_reuse_valid_script_hash_as_caption_proof(self):
        self.doc['scenes'][0]['source_text'] = 'Текст, которого нет в субтитрах.'
        self.save()
        with self.assertRaisesRegex(Blocked, 'source words'):
            validate_project(self.project, smoke=True)

    def test_retaken_voice_is_not_masked_by_a_same_duration_master_cache(self):
        from studio.story_soundtrack import normalize_voice
        mastered = self.root / 'master.wav'
        normalize_voice(self.voice, mastered, 1.)
        with self.voice.open('r+b') as stream:
            stream.seek(100); stream.write(b'\x00\x00')
        with self.assertRaisesRegex(ValueError, 'source or cached WAV changed'):
            normalize_voice(self.voice, mastered, 1.)

    def test_encoded_master_is_measured_and_native_speech_clock_is_preserved(self):
        result = run(self.project, self.root / 'out', smoke=True)
        levels = result['qa']['av_delivery']['encoded_audio']
        self.assertAlmostEqual(levels['integrated_lufs'], -16., delta=1.)
        self.assertLessEqual(levels['true_peak_dbtp'], -1.)
        self.assertEqual(result['qa']['audio_mastering']['speech_speed_factor'], 1.)
        self.assertAlmostEqual(result['qa']['av_delivery']['video_clock_seconds'], 2., delta=.001)

    def test_thumbnail_cannot_be_silently_upscaled_into_a_master(self):
        self.doc.update(width=1080,height=1920)
        self.doc['scenes']=[{**self.doc['scenes'][0],'id':i} for i in range(1,9)]
        self.save()
        with self.assertRaisesRegex(Blocked, 'thumbnail'):
            validate_project(self.project)

    def test_cover_uses_rendered_focal_composition_not_a_separate_center_crop(self):
        image=Image.new('RGB',(960,540),(35,75,110))
        ImageDraw.Draw(image).rectangle((800,180,900,350),fill=(240,25,25))
        image.save(self.root/'frame.png')
        self.doc['scenes'][0]['framing']={'source_focus':[.88,.49]}
        self.doc['cover_text']=''
        self.save()
        result=run(self.project,self.root/'out',smoke=True)
        with Image.open(self.root/'out/upgrade_fixture/cover.jpg') as source:
            rgb=np.asarray(source.convert('RGB'))
        self.assertGreater(int(((rgb[:,:,0]>160)&(rgb[:,:,1]<70)).sum()),1000)
        self.assertTrue(result['qa']['cover_source']['encoded_final_frame'])

    def test_changed_music_cannot_reuse_a_same_duration_story_mix(self):
        from studio.story_soundtrack import mix_tracks
        dest=self.root/'mix.wav'
        mix_tracks(self.voice,self.voice,self.voice,dest,1.)
        self.assertEqual(mix_tracks(self.voice,self.voice,self.voice,dest,1.)['sha256'],
                         hashlib.sha256(dest.read_bytes()).hexdigest())
        with self.voice.open('r+b') as stream:
            stream.seek(100);stream.write(b'\x00\x00')
        with self.assertRaisesRegex(ValueError, 'Mix source or cached WAV changed'):
            mix_tracks(self.voice,self.voice,self.voice,dest,1.)

    def test_remote_source_bundle_can_render_without_the_host_repository(self):
        import io
        from studio.kaggle_shorts import build_source
        repository=Path(__file__).resolve().parents[1]
        isolated=self.root/'isolated';isolated.mkdir()
        with zipfile.ZipFile(io.BytesIO(build_source(repository))) as archive:
            archive.extractall(isolated)
        project=isolated/'project';project.mkdir()
        for name in ['project.json','story.txt','voice.wav','frame.png']:
            shutil.copy2(self.root/name,project/name)
        environment={**os.environ,'PYTHONPATH':str(isolated)}
        result=subprocess.run([sys.executable,'-m','studio.shorts_factory','--project',str(project/'project.json'),
                               '--out',str(isolated/'output'),'--technical-smoke'],
                              cwd=isolated,env=environment,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=json.loads(result.stdout)
        self.assertTrue(report['qa']['av_delivery']['constant_frame_rate_verified'])
        self.assertTrue((isolated/'studio/voice_audit.py').is_file())

    def test_changed_scene_bed_cannot_hide_behind_an_old_aggregate_music_file(self):
        from studio.story_soundtrack import crossfade_beds
        target=self.root/'combined_music.wav'
        crossfade_beds([self.voice],[1.],target,fade=.2)
        self.assertEqual(crossfade_beds([self.voice],[1.],target,fade=.2),'')
        with self.voice.open('r+b') as stream:
            stream.seek(100);stream.write(b'\x00\x00')
        with self.assertRaisesRegex(ValueError,'Assembled audio source'):
            crossfade_beds([self.voice],[1.],target,fade=.2)

    def test_narration_assembly_checks_the_changed_take_even_when_duration_matches(self):
        from studio.story_soundtrack import concat_voices
        target=self.root/'combined_narration.wav'
        concat_voices([self.voice],[1.],target)
        concat_voices([self.voice],[1.],target)
        with self.voice.open('r+b') as stream:
            stream.seek(100);stream.write(b'\x00\x00')
        with self.assertRaisesRegex(ValueError,'Assembled audio source'):
            concat_voices([self.voice],[1.],target)


if __name__ == '__main__':
    unittest.main()
