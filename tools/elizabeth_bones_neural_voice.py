#!/usr/bin/env python3
"""Elizabeth Bones EP01: real Russian neural voice tracks, free-first, review only.
Outputs separately authored lines + a dialogue-only WAV matching EP01 V3 SRT timestamps.
No publication, no keys or credentials, no fake voice fallback.
"""
import asyncio
import json
import subprocess
import time
from pathlib import Path
from pydub import AudioSegment
import os

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT / "dist" / "elizabeth_bones_ru_voice"
DEST.mkdir(parents=True,exist_ok=True)
LINES=[
 ("ЭЛИЗАБЕТ","Следующий. Имя?",2.200,3.766),
 ("КОСТЯ","Константин Мортисович Костин. Для друзей — Костя.",4.336,7.997),
 ("ЭЛИЗАБЕТ","Возраст?",8.567,9.587),
 ("КОСТЯ","Двести три. Но душой я мальчишка!",10.157,12.632),
 ("ЭЛИЗАБЕТ","Вижу. А где душа?",13.202,14.969),
 ("КОСТЯ","На испытательном. Уже три века.",15.539,17.866),
 ("ЭЛИЗАБЕТ","Опыт работы?",18.436,19.676),
 ("КОСТЯ","Двести лет без больничных. И всё ещё в отличной форме!",20.246,23.968),
 ("ЭЛИЗАБЕТ","А почему уволились?",24.538,25.889),
 ("КОСТЯ","Коллектив развалился. В буквальном смысле.",26.459,29.479),
 ("ЭЛИЗАБЕТ","У нас не шутят над чужой болью.",30.049,32.318),
 ("КОСТЯ","Я знаю. С семьями — только уважение.",32.888,35.914),
 ("ЭЛИЗАБЕТ","Удивили. И что вы умеете?",36.484,38.738),
 ("КОСТЯ","Выслушать. Поддержать. И молчать, когда нужно.",39.308,43.145),
 ("ЭЛИЗАБЕТ","Вы приняты. Но за вами нужен глаз да глаз.",43.715,46.943),
 ("КОСТЯ","Договорились. И, Элизабет... Можно просто Лиз?",47.513,50.783),
 ("ЭЛИЗАБЕТ","Только не на работе, Костя.",51.353,53.494),
 ("КОСТЯ","А после работы — кофе?",54.064,55.921),
 ("ЭЛИЗАБЕТ","Сначала научись держать чашку.",56.491,58.628),
 ("КОСТЯ","Не волнуйся. Руки-то у меня золотые! Почти.",59.198,62.669),
 ("ЭЛИЗАБЕТ","Ладно. Кажется, мы сработаемся.",63.239,66.129),
]
VOICES={"ЭЛИЗАБЕТ":("ru_RU-irina-medium","offline"),"КОСТЯ":("ru_RU-dmitri-medium","offline")}
async def generate_line(idx,speaker,text):
    model_name=VOICES[speaker][0]
    model=ROOT / "dist" / "models" / (model_name+".onnx")
    if not model.is_file():
        raise FileNotFoundError("Piper model must be downloaded before synthesis: "+str(model))
    target=DEST / f"line_{idx:02d}_{'liz' if speaker=='ЭЛИЗАБЕТ' else 'kostya'}.source.wav"
    subprocess.run(["piper","--model",str(model),"--output_file",str(target)],
                   input=text+"\n",text=True,check=True,capture_output=True)
    if not target.exists() or target.stat().st_size<1000:
        raise RuntimeError("Neural TTS produced empty audio")
    return target
async def main():
    combined=AudioSegment.silent(duration=70300,frame_rate=48000).set_channels(2)
    metadata=[]
    for i,(speaker,content,start,end) in enumerate(LINES,1):
        raw=await generate_line(i,speaker,content)
        slot=(end-start)*1000
        source=AudioSegment.from_file(raw)
        original_ms=len(source)
        factor=max(1.0,original_ms/max(slot-60,200))
        wav=DEST / f"line_{i:02d}_{'liz' if speaker=='ЭЛИЗАБЕТ' else 'kostya'}.wav"
        command=["ffmpeg","-nostdin","-y","-loglevel","error","-i",str(raw),
                 "-af",f"atempo={factor:.5f}","-ar","48000","-ac","2",str(wav)]
        # atempo supports 0.5..100 in modern FFmpeg, normal lines usually <= 1.5x
        subprocess.run(command,check=True)
        clip=AudioSegment.from_wav(wav)
        if len(clip) > slot + 80:
            raise RuntimeError(f"Line {i} exceeds slot: {len(clip)} > {slot} ms")
        combined=combined.overlay(clip,position=int(round(start*1000)))
        metadata.append({"line":i,"speaker":speaker,"voice":VOICES[speaker][0],
                         "text":content,"start":start,"end":end,
                         "source_ms":original_ms,"adjusted_ms":len(clip),
                         "tempo_factor":round(factor,3)})
        print(f"[{i:02d}/{len(LINES)}] {speaker} {len(clip)} ms / {slot:.0f} ms; speed {factor:.2f}x",flush=True)
        await asyncio.sleep(0.35)
    combined.export(DEST/"EP01_RU_NEURAL_DIALOGUE_70s.wav",format="wav")
    subprocess.run(["ffmpeg","-nostdin","-y","-loglevel","error","-i",
        str(DEST/"EP01_RU_NEURAL_DIALOGUE_70s.wav"),"-c:a","aac","-b:a","192k",
        str(DEST/"EP01_RU_NEURAL_DIALOGUE_70s.m4a")],check=True)
    (DEST/"manifest.json").write_text(json.dumps({"schema":1,
        "status":"generated_review_required","language":"ru","duration_ms":70300,
        "voices":VOICES,"lines":metadata},ensure_ascii=False,indent=2),encoding="utf-8")
    (DEST/"HOW_TO_MUX.txt").write_text(
        "EP01 V3: copy the source MP4 and this M4A into the same directory. "
        "Then run (removes placeholder eSpeak and music):\n"
        "ffmpeg -i ELIZABETH_BONES_EP01_V3_FULL_VOICED.mp4 "
        "-i EP01_RU_NEURAL_DIALOGUE_70s.m4a -map 0:v:0 -map 1:a:0 "
        "-c:v copy -c:a aac -b:a 192k -shortest ELIZABETH_BONES_EP01_V4_RU.mp4\n"
        "Generated preview voices require hearing review. "
        "Use separately exported individual WAVs if adjusting emotional timing.\n",
        encoding="utf-8")
    print("VOICE PASS: 21 lines, WAV+M4A ready; do not claim human-reviewed.",flush=True)
if __name__=="__main__":
    asyncio.run(main())
