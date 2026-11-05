"""Private, scene-aware review of the encoded short; no automatic approval.

The viewer is local HTML, uses only copied media and needs no server or API.
It never changes the film, its screenplay or a publication schedule.
"""
from __future__ import annotations

import html
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def _ffmpeg(args: list[str]) -> None:
    result = subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin',
                             '-n', *args], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Review media failed: ' + result.stderr[-1500:])


def timeline(entries: list[dict], fps: int) -> dict:
    """Derive every boundary from the same integer frame clock as the editor."""
    from .shorts_factory import frame_count, sha
    if not entries or type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError('Review needs real scenes and an integer frame clock')
    scenes, offset = [], 0
    for entry in entries:
        seconds = entry['seconds']
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Review scene duration is invalid')
        frames = frame_count(seconds, fps)
        start, end = offset / fps, (offset + frames) / fps
        cues = [{'start_s': start + c['start'],
                 'end_s': min(end, start + c['end']), 'text': c['text']}
                for c in entry['cues']]
        if any(c['start_s'] >= c['end_s'] for c in cues):
            raise ValueError('Caption has no visible interval in its encoded scene')
        scenes.append({'id': entry['id'], 'start_frame': offset,
                       'end_frame_exclusive': offset + frames,
                       'start_s': start, 'end_s': end,
                       'sample_frames': sorted({offset, offset + frames // 2, offset + frames - 1}),
                       'source_image_sha256': sha(entry['image']),
                       'source_voice_sha256': sha(entry['voice']),
                       'camera': entry.get('camera', 'push_in'),
                       'caption_position': entry.get('caption_position', 'bottom'),
                       'cues': cues})
        offset += frames
    return {'schema': 1, 'fps': fps, 'total_frames': offset,
            'duration_s': offset / fps, 'scenes': scenes,
            'source': 'encoded_final_mp4', 'artistic_approval': False}


def join_narration(entries: list[dict], output: Path, fps: int) -> None:
    """Preserve original speech, pad only scene tails, and export lossless WAV."""
    from .shorts_factory import frame_count
    if output.exists():
        raise FileExistsError('Do not overwrite a narration export')
    args, filters, expected = [], [], 0
    for index, scene in enumerate(entries):
        count = round(frame_count(scene['seconds'], fps) * 48000 / fps)
        args += ['-i', str(scene['voice'])]
        filters.append(f'[{index}:a:0]aresample=48000,aformat=channel_layouts=stereo,'
                       f'apad=whole_len={count},atrim=end_sample={count},asetpts=N/SR/TB[a{index}]')
        expected += count
    if not entries:
        raise ValueError('No original narration scenes')
    filters.append(''.join(f'[a{i}]' for i in range(len(entries))) +
                   f'concat=n={len(entries)}:v=0:a=1[out]')
    _ffmpeg([*args, '-filter_complex', ';'.join(filters), '-map', '[out]',
             '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le', str(output)])
    import wave
    with wave.open(str(output), 'rb') as wav:
        if wav.getnframes() != expected:
            raise RuntimeError('Narration WAV differs from the scene sample clock')


def _extract_frames(video: Path, folder: Path, frames: list[int]) -> dict[int, Path]:
    """One decode pass, selected by decoded frame index rather than rounded seek."""
    expression = '+'.join(f'eq(n,{n})' for n in frames)
    _ffmpeg(['-i', str(video), '-map', '0:v:0', '-an', '-vf',
             f"select='{expression}',setsar=1",
             '-fps_mode', 'vfr', '-start_number', '0', '-q:v', '3',
             str(folder / 'frame_%04d.jpg')])
    paths = sorted(folder.glob('frame_*.jpg'))
    if len(paths) != len(frames):
        raise RuntimeError('Encoded MP4 does not contain all planned review frames')
    return dict(zip(frames, paths))


def _board(source: Path | None, frame_numbers: list[int], decoded: dict[int, Path],
           output: Path, fps: int) -> None:
    from .caption_layout import font
    tiles = ([('Исходник', source)] if source else []) + [
        (f'MP4 · {n / fps:.3f} с', decoded[n]) for n in frame_numbers]
    gap, width, height = 8, 180, 320
    board = Image.new('RGB', (gap + len(tiles) * (width + gap), height + 42), '#101a28')
    draw = ImageDraw.Draw(board)
    for index, (label, path) in enumerate(tiles):
        with Image.open(path) as im:
            # Contain the entire source; another center crop would hide lost props.
            thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),
                                     (width, height), Image.Resampling.LANCZOS)
        x = gap + index * (width + gap)
        board.paste(thumb, (x + (width - thumb.width) // 2, (height - thumb.height) // 2))
        draw.text((x + 2, height + 10), label, fill='white', font=font(14))
    board.save(output, quality=92)


def render_html(data: dict, title: str) -> str:
    """Keep story text inert, including literal closing-script or HTML tags."""
    payload = json.dumps(data, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c')
    return re.sub('__TITLE__|__DATA__',
                  lambda m: html.escape(title) if m[0] == '__TITLE__' else payload, _HTML)


CORE_FILES = {'final.mp4', 'cover.jpg', 'subtitles.srt', 'qa_report.json'}
REVIEW_FILES = CORE_FILES | {'subtitles.ass', 'script_ru.txt', 'narration_ru.wav',
                              'audio_master.wav', 'timeline.json', 'review.html',
                              'review_assets/opening.jpg'}


def delivery_files(folder: Path) -> list[str]:
    """Verify all modern deliverables; old five-file renders remain readable."""
    from .shorts_factory import sha
    folder = Path(folder).resolve()
    receipt_path = folder / 'receipt.json'
    if receipt_path.is_symlink():
        raise ValueError('Delivery receipt must be a real local file')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt.get('approved_for_publication') is not False or
            receipt.get('final_sha256') != sha(folder / 'final.mp4')):
        raise ValueError('Delivery video checksum or review state changed')
    checksums = receipt.get('delivery_sha256')
    if checksums is None and str(receipt.get('engine', '')).startswith('1.4.'):
        raise ValueError('Modern delivery is missing its file checksums')
    if checksums is not None:
        if not isinstance(checksums, dict) or not REVIEW_FILES.issubset(checksums):
            raise ValueError('Incomplete modern delivery checksum manifest')
        names = sorted(checksums)
    else:
        names = sorted(CORE_FILES)
    for name in names:
        if (not isinstance(name, str) or name not in REVIEW_FILES and not
                re.fullmatch(r'review_assets/scene_[0-9]{3}(?:_detail)?\.jpg', name)):
            raise ValueError('Unexpected or unsafe delivery file')
        path = folder / name
        if (not path.is_file() or path.is_symlink() or
                folder not in path.resolve().parents):
            raise ValueError('Missing or escaped delivery file')
        if checksums is not None and checksums[name] != sha(path):
            raise ValueError(f'Delivery checksum changed: {name}')
    qa = json.loads((folder / 'qa_report.json').read_text(encoding='utf-8'))
    if (qa.get('technical_pass') is not True or qa.get('full_decode_pass') is not True
            or qa.get('approved_for_publication') is not False):
        raise ValueError('Delivery technical QA is missing or failed')
    if checksums is not None:
        data = json.loads((folder / 'timeline.json').read_text(encoding='utf-8'))
        expected_boards = {s[key] for s in data['scenes'] for key in ('board','detail')}
        listed_boards = {name for name in names if re.fullmatch(r'review_assets/scene_[0-9]{3}(?:_detail)?\.jpg', name)}
        if (not expected_boards or expected_boards != listed_boards or
                data.get('final_sha256') != receipt['final_sha256'] or
                data.get('script_sha256') != receipt.get('script_sha256') or
                sha(folder / 'script_ru.txt') != receipt.get('script_sha256')):
            raise ValueError('Review timeline, boards or original script differs from the film receipt')
    return ['receipt.json', *names]


def write_review_archive(folders: list[Path], output: Path, *, prefix: str = '') -> dict:
    """Publish one complete ZIP atomically; hash the bytes actually archived."""
    import hashlib
    import os
    import zipfile
    from .shorts_factory import sha
    if not folders or len({p.name for p in folders}) != len(folders):
        raise ValueError('Delivery needs existing uniquely named story folders')
    if prefix not in ('', 'review'):
        raise ValueError('Unsupported review archive prefix')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Do not overwrite a previous delivery archive')
    assets = []
    for folder in folders:
        if folder.is_symlink() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', folder.name):
            raise ValueError('Unexpected delivery directory')
        receipt_bytes = (folder / 'receipt.json').read_bytes()
        receipt = json.loads(receipt_bytes)
        for name in delivery_files(folder):
            path = folder / name
            if name == 'receipt.json':
                expected = hashlib.sha256(receipt_bytes).hexdigest()
            elif 'delivery_sha256' in receipt:
                expected = receipt['delivery_sha256'][name]
            elif name == 'final.mp4':
                expected = receipt['final_sha256']
            else:
                expected = sha(path)
            assets.append((path, '/'.join(x for x in (prefix, folder.name, name) if x), expected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.review-archive-', suffix='.zip', delete=False) as temp:
        staging = Path(temp.name)
    try:
        with zipfile.ZipFile(staging, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for path, name, expected in assets:
                digest = hashlib.sha256()
                with path.open('rb') as source, archive.open(name, 'w') as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk)
                        target.write(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError('Delivery changed during archive creation')
        with staging.open('rb') as source:
            os.fsync(source.fileno())
        os.link(staging, output)
    finally:
        staging.unlink(missing_ok=True)
    return {'status': 'REVIEW_BUNDLE_ONLY', 'videos': len(folders),
            'files': len(assets), 'zip': str(output), 'published': False}


def create_review(item: dict, folder: Path) -> dict:
    import shutil
    from .shorts_factory import probe, sha
    video = folder / 'final.mp4'
    digest = sha(video)
    data = timeline(item['entries'], item['fps'])
    stream = next(s for s in probe(video)['streams'] if s['codec_type'] == 'video')
    if int(stream.get('nb_frames', -1)) != data['total_frames']:
        raise RuntimeError('Review timeline differs from the encoded frame count')
    data.update(story_id=item['id'], title=item['title'],
                final_sha256=digest, script_sha256=item['digest'],
                project_sha256=item['project_sha256'])
    output = folder / 'review_assets'
    output.mkdir(exist_ok=False)
    opening = sorted({min(data['total_frames'] - 1, round(t * item['fps']))
                      for t in (0, .5, 1, 2 - 1 / item['fps'])})
    all_frames = sorted(set(opening + [n for scene in data['scenes'] for n in scene['sample_frames']]))
    with tempfile.TemporaryDirectory(prefix='tw-review-frames-') as temp:
        decoded = _extract_frames(video, Path(temp), all_frames)
        _board(None, opening, decoded, output / 'opening.jpg', item['fps'])
        for index, (entry, scene) in enumerate(zip(item['entries'], data['scenes']), 1):
            scene['board'] = f'review_assets/scene_{index:03d}.jpg'
            scene['detail'] = f'review_assets/scene_{index:03d}_detail.jpg'
            _board(entry['image'], scene['sample_frames'], decoded,
                   folder / scene['board'], item['fps'])
            middle = scene['sample_frames'][len(scene['sample_frames']) // 2]
            shutil.copyfile(decoded[middle],folder / scene['detail'])
    if sha(video) != digest:
        raise RuntimeError('Encoded video changed while review frames were decoded')
    (folder / 'timeline.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (folder / 'review.html').write_text(render_html(data, item['title']), encoding='utf-8')
    return {'viewer': 'review.html', 'timeline': 'timeline.json',
            'frame_selection': 'decoded_integer_frame_indices',
            'source': 'encoded_final_mp4', 'artistic_approval': False}


_HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>__TITLE__ · STORY GLITCH</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101723;color:#eef3fc}
*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:22px}
h1{font-size:24px;margin:0 0 10px}p{line-height:1.55;color:#b9c7da}
a{color:#94d4ff}button{background:#243a55;color:#edf6ff;border:1px solid #537699;border-radius:7px;padding:9px 13px;cursor:pointer;font:inherit;text-align:left}
button:hover,button:focus-visible{background:#365775;outline:2px solid #a2ddff}button[aria-current=true]{border-color:#f2b57b}
.layout{display:grid;grid-template-columns:minmax(240px,350px) 1fr;gap:24px;align-items:start}
.player{position:sticky;top:16px}.screen{position:relative;aspect-ratio:9/16;max-height:73vh;max-width:calc(73vh * 9 / 16);margin:auto}video{display:block;width:100%;height:100%;background:#000;object-fit:contain}
.safe{pointer-events:none;display:none;position:absolute;inset:18% 14.45% 16.67% 7.04%;border:2px dashed #ffba74}
.safe.visible{display:block}.tools{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.muted{font-size:13px;color:#a6b9d0}
article{border:1px solid #30465f;border-radius:10px;padding:15px;margin-bottom:17px}article h2{font-size:17px;margin:0 0 10px}
img{display:block;width:100%;height:auto;border-radius:5px;background:#101a28}.cues{display:grid;gap:7px;margin-top:12px}
.cue{font-size:14px;line-height:1.4}label{display:block;margin-top:12px;font-size:14px;color:#bfcee2}textarea{width:100%;min-height:65px;margin-top:7px;background:#111f30;color:#f1f6ff;border:1px solid #537699;border-radius:6px;padding:10px;font:inherit}
#error{color:#ffd09b}.identity{overflow-wrap:anywhere;font:12px monospace;color:#a7bbd3}
@media(max-width:760px){main{padding:14px}.layout{display:block}.player{position:static;max-width:350px;margin:0 auto 24px}h1{font-size:21px}}
</style></head><body><main>
<h1>__TITLE__</h1><p>STORY GLITCH · просмотр готового MP4 по сценам. Сравните лица, руки, предметы, субтитры и стыки; прослушайте речь и окончание. Замечания сохраняются отдельным файлом.</p>
<div class="layout"><section class="player" aria-label="Видеопросмотр">
<div class="screen"><video id="movie" controls playsinline preload="metadata" poster="cover.jpg"><source src="final.mp4" type="video/mp4"></video><div class="safe" id="safe"></div></div>
<div class="tools"><button id="opening">Первые 2 секунды</button><button id="safeButton" aria-pressed="false">Зона текста</button></div>
<p class="muted">Контур — ориентир композиции. Реальное отображение интерфейса и обложки проверяется после загрузки в TikTok Studio.</p>
<div class="tools"><a href="subtitles.srt" download>Субтитры</a><a href="script_ru.txt" download>Сценарий</a><a href="narration_ru.wav" download>Диктор WAV</a><a href="audio_master.wav" download>Мастер WAV</a></div>
<button id="save">Скачать замечания</button><p id="error" role="status"></p><p class="identity" id="identity"></p>
</section><section aria-label="Сцены и реплики"><article><h2>Начало: 0–2 секунды</h2><img src="review_assets/opening.jpg" alt="Фактические кадры начала MP4"></article><div id="scenes"></div></section></div>
</main><script id="reviewData" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('reviewData').textContent);
const movie=document.getElementById('movie'), error=document.getElementById('error');
let until=null, animation=null;
const buttons=[], notes=[];
function checkEnd(){if(until!==null && movie.currentTime>=until){movie.pause();until=null;}else if(!movie.paused){animation=requestAnimationFrame(checkEnd);}}
function playRange(start,end){cancelAnimationFrame(animation);until=end;movie.currentTime=start;error.textContent='';movie.play().then(()=>{cancelAnimationFrame(animation);animation=requestAnimationFrame(checkEnd);}).catch(()=>{error.textContent='Нажмите воспроизведение в плеере, чтобы прослушать выбранный фрагмент.';});}
function makeButton(text,start,end){const b=document.createElement('button');b.textContent=text;b.type='button';b.dataset.start=start;b.dataset.end=end;b.onclick=()=>playRange(start,end);buttons.push(b);return b;}
const root=document.getElementById('scenes');
data.scenes.forEach((scene,index)=>{const article=document.createElement('article');
 const heading=document.createElement('h2');heading.textContent=`Сцена ${scene.id} · ${scene.start_s.toFixed(2)}–${scene.end_s.toFixed(2)} с`;article.appendChild(heading);
 const image=document.createElement('img');image.src=scene.board;image.loading='lazy';image.alt=`Сцена ${scene.id}: полный исходник, начало, середина и конец MP4`;article.appendChild(image);
 const detail=document.createElement('a');detail.href=scene.detail;detail.target='_blank';detail.rel='noopener';detail.textContent='Кадр MP4 в полном размере';article.appendChild(detail);
 const tools=document.createElement('div');tools.className='tools';tools.appendChild(makeButton('Смотреть сцену',scene.start_s,scene.end_s));
 if(index>0)tools.appendChild(makeButton('Проверить стык',Math.max(0,scene.start_s-.5),Math.min(data.duration_s,scene.start_s+.7)));
 article.appendChild(tools);const cues=document.createElement('div');cues.className='cues';
 scene.cues.forEach(cue=>{const b=makeButton(`${cue.start_s.toFixed(2)} · ${cue.text}`,cue.start_s,Math.min(data.duration_s,cue.end_s+.12));b.className='cue';cues.appendChild(b);});article.appendChild(cues);
 const label=document.createElement('label');label.textContent='Замечания к сцене';const note=document.createElement('textarea');note.setAttribute('aria-label',`Замечания к сцене ${scene.id}`);label.appendChild(note);notes.push({scene:scene.id,note});article.appendChild(label);root.appendChild(article);
});
document.getElementById('opening').onclick=()=>playRange(0,Math.min(2,data.duration_s));
movie.addEventListener('timeupdate',()=>{buttons.forEach(b=>b.setAttribute('aria-current',String(movie.currentTime>=Number(b.dataset.start)&&movie.currentTime<Number(b.dataset.end))));});
movie.addEventListener('pause',()=>cancelAnimationFrame(animation));
movie.addEventListener('play',()=>{cancelAnimationFrame(animation);animation=requestAnimationFrame(checkEnd);});
document.getElementById('safeButton').onclick=function(){const visible=document.getElementById('safe').classList.toggle('visible');this.setAttribute('aria-pressed',String(visible));};
document.getElementById('identity').textContent=`MP4 SHA-256: ${data.final_sha256}`;
document.getElementById('save').onclick=()=>{const report={schema:1,story_id:data.story_id,final_sha256:data.final_sha256,script_sha256:data.script_sha256,project_sha256:data.project_sha256,created_at:new Date().toISOString(),notes:notes.map(x=>({scene:x.scene,text:x.note.value})),artistic_approval:false};
 const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='review_notes.json';document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></body></html>'''
