"""Generate full Elizabeth Bones EP01 with Qwen3-TTS, two stable voices, natural durations.
No robotic tempo forcing, no release, no fallback to Piper. Use Github artifact for QA.
"""
from __future__ import annotations
import json
from pathlib import Path
import soundfile as sf
import numpy as np
from studio.russian_qwen_tts import synthesize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"dist"/"elizabeth_bones_qwen_v5"
OUT.mkdir(parents=True,exist_ok=True)
LINES=[
("Элизабет","Следующий. Имя?"),
("Костя","Константин Мортисович Костин. Для друзей — Костя."),
("Элизабет","Возраст?"),
("Костя","Двести три. Но душой я мальчишка!"),
("Элизабет","Вижу. А где душа?"),
("Костя","На испытательном. Уже три века."),
("Элизабет","Опыт работы?"),
("Костя","Двести лет без больничных. И всё ещё в отличной форме!"),
("Элизабет","А почему уволились?"),
("Костя","Коллектив развалился. В буквальном смысле."),
("Элизабет","У нас не шутят над чужой болью."),
("Костя","Я знаю. С семьями — только уважение."),
("Элизабет","Удивили. И что вы умеете?"),
("Костя","Выслушать. Поддержать. И молчать, когда нужно."),
("Элизабет","Вы приняты. Но за вами нужен глаз да глаз."),
("Костя","Договорились. И, Элизабет... Можно просто Лиз?"),
("Элизабет","Только не на работе, Костя."),
("Костя","А после работы — кофе?"),
("Элизабет","Сначала научись держать чашку."),
("Костя","Не волнуйся. Руки-то у меня золотые! Почти."),
("Элизабет","Ладно. Кажется, мы сработаемся."),
]
PROFILES={"Элизабет":("Serena","warm"),"Костя":("Ryan","joyful")}
def main():
    meta=[]; samples=[]; rate=None
    for number,(who,words) in enumerate(LINES,1):
        voice,style=PROFILES[who]
        target=OUT/f"{number:02d}_{'elizabeth' if who=='Элизабет' else 'kostya'}.wav"
        info=synthesize(words,voice,target,style=style)
        wav,sr=sf.read(target,dtype="float32",always_2d=False)
        if wav.ndim>1: raise RuntimeError("Unexpected multi-channel TTS")
        if rate is None:rate=sr
        if sr!=rate:raise RuntimeError("Inconsistent rates")
        # Natural pauses. Never speed up / time stretch speech.
        samples.append(np.asarray(wav,dtype="float32"))
        samples.append(np.zeros(int(sr*.50),dtype="float32"))
        meta.append({"i":number,"speaker":who,"text":words,"voice":voice,"style":style,
                     "duration":round(len(wav)/sr,3),"file":target.name})
        print(f"{number:02d}/21 {who}: {len(wav)/sr:.2f}s",flush=True)
    combined=np.concatenate(samples).astype("float32")
    sf.write(OUT/"EP01_V5_QWEN_NATURAL_DIALOGUE.wav",combined,rate,subtype="PCM_16")
    (OUT/"manifest.json").write_text(json.dumps({"status":"needs_human_listening_review",
        "voices":PROFILES,"language":"ru","speech_speed_forced":False,
        "total_duration":len(combined)/rate,"lines":meta},ensure_ascii=False,indent=2),encoding="utf-8")
    assert len(meta)==21
    print("QWEN NATURAL VOICE READY",len(combined)/rate,flush=True)

if __name__=="__main__": main()
