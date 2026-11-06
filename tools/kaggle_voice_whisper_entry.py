"""Independent multilingual speech audit on CPU; never changes approved text."""
from __future__ import annotations
import base64,io,json,os,subprocess,sys,time,zipfile
from pathlib import Path

CONFIG_B64='__TW_VOICE_QA_CONFIG__'
SOURCE_B64='__TW_VOICE_QA_SOURCE__'
OUT=Path('/kaggle/working');INPUT=Path('/kaggle/input');BASE=OUT/'tw_voice_whisper_modules'


def main():
    conf=json.loads(base64.b64decode(CONFIG_B64));BASE.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:
        for item in z.infolist():
            p=Path(item.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!='studio':raise ValueError('Unsafe source bundle')
        z.extractall(BASE)
    sys.path.insert(0,str(BASE));os.environ['HF_HUB_DISABLE_XET']='1'
    subprocess.run([sys.executable,'-m','pip','install','-q','faster-whisper'],check=True,timeout=1200)
    from faster_whisper import WhisperModel
    from studio.kaggle_shorts_worker import _safe_extract
    from studio.autonomous_factory import atomic_json
    import re,wave,numpy as np
    def words(text):return re.findall(r'[а-яa-z0-9]+',text.lower().replace('ё','е'))
    def distance(a,b):
        previous=list(range(len(b)+1))
        for i,left in enumerate(a,1):
            current=[i]
            for j,right in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
            previous=current
        return previous[-1]
    selected=conf.get('qa_scenes')
    if selected is not None and (not isinstance(selected,dict) or not selected or
        any(k not in {str(i) for i in range(13,19)} or not isinstance(v,list) or not v or
            any(type(i) is not int or i not in range(1,9) for i in v) for k,v in selected.items())):
        raise ValueError('Speech recheck must name bounded original scene IDs')
    cached=[p.parent for p in INPUT.rglob('model.bin') if conf.get('model_kernel_slug') in p.parts]
    model=WhisperModel(str(cached[0]) if len(cached)==1 else 'small',device='cpu',compute_type='int8',cpu_threads=4,
                       download_root=str(OUT/'private_models/independent_whisper'))
    state={'schema':1,'status':'RUNNING','model':'Systran/faster-whisper-small',
           'gpu_used':False,'source_text_supplied_to_decoder':False,'stories':{},
           'artistic_listening_review':'not claimed','published':False,'paid_services':False}
    started=time.monotonic()
    for sid in range(13,19):
        if selected is not None and str(sid) not in selected:continue
        archives=[p for p in INPUT.rglob(f'TwoWorlds_PRIVATE_{sid}_voice.zip') if conf['voice_kernel_slug'] in p.parts]
        if len(archives)!=1:raise FileNotFoundError('Original full voice recovery archive missing')
        root=OUT/f'voice_inputs/{sid}'
        with zipfile.ZipFile(archives[0]) as z:_safe_extract(z,root)
        project=root/'production_queue'/str(sid);doc=json.loads((project/'project.json').read_text())
        shots={s['scene']:s for s in json.loads((project/'shot_plan.json').read_text())};results=[]
        for scene in doc['scenes']:
            if selected is not None and scene['id'] not in selected[str(sid)]:continue
            # Decode our own known PCM WAVs; avoid an installed PyAV ABI/API
            # mismatch on the CPU image and retain their actual sample clock.
            with wave.open(str(project/scene['voice']),'rb') as wav:
                if wav.getsampwidth()!=2 or wav.getnchannels()!=1:raise ValueError('Expected original mono PCM16 narration')
                rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(np.float32)/32768
            samples=np.interp(np.arange(round(len(pcm)*16000/rate))*rate/16000,np.arange(len(pcm)),pcm).astype(np.float32)
            observed=[];spans=[]
            cursor=0;sr=16000
            while cursor<len(samples):
                if len(samples)-cursor<=24*sr:end=len(samples)
                else:
                    start=cursor+12*sr;stop=cursor+20*sr;step=800
                    energy=[float(np.mean(samples[i:i+step]**2)) for i in range(start,stop,step)]
                    end=start+int(np.argmin(energy))*step+step//2
                segments,info=model.transcribe(samples[cursor:end],language='ru',task='transcribe',
                    beam_size=5,temperature=(0,.2,.4),no_repeat_ngram_size=3,
                    condition_on_previous_text=False,word_timestamps=True,vad_filter=False)
                offset=cursor/sr
                for segment in segments:
                    observed.append(segment.text)
                    spans.append({'start':offset+segment.start,'end':offset+segment.end,'text':segment.text,
                        'average_log_probability':segment.avg_logprob,'no_speech_probability':segment.no_speech_prob,
                        'words':[{'word':w.word,'start':offset+w.start,'end':offset+w.end,'probability':w.probability} for w in segment.words or []]})
                cursor=end
            transcript=' '.join(observed);expected=shots[scene['id']]['exact_source_text']
            reference=words(expected);recognized=words(transcript);wer=distance(reference,recognized)/max(1,len(reference))
            results.append({'scene':scene['id'],'reference':expected,'recognized':transcript,
                'word_error_rate':round(wer,4),'requires_review':wer>.12,'segments':spans,'source_text_changed':False})
            state['stories'][str(sid)]=results;state['elapsed_s']=round(time.monotonic()-started,2)
            atomic_json(OUT/'batch_voice_whisper_qa.json',state)
        print('Completed independent CPU Whisper speech audit for story',sid,flush=True)
    state['status']='COMPLETE'
    atomic_json(OUT/'batch_voice_whisper_qa.json',state)


if __name__=='__main__':main()
