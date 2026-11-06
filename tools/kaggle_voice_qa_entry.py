"""Independent, read-only Russian speech transcription audit on CPU."""
from __future__ import annotations
import base64,io,json,os,re,subprocess,sys,time,wave,zipfile
from pathlib import Path

CONFIG_B64='__TW_VOICE_QA_CONFIG__'
SOURCE_B64='__TW_VOICE_QA_SOURCE__'
OUT=Path('/kaggle/working');INPUT=Path('/kaggle/input');BASE=OUT/'tw_voice_qa_modules'


def words(text):return re.findall(r'[а-яa-z0-9]+',text.lower().replace('ё','е'))


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,left in enumerate(a,1):
        current=[i]
        for j,right in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]


def main():
    conf=json.loads(base64.b64decode(CONFIG_B64));BASE.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:
        for item in z.infolist():
            p=Path(item.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!='studio':raise ValueError('Unsafe source bundle')
        z.extractall(BASE)
    sys.path.insert(0,str(BASE))
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers','safetensors'],check=True,timeout=1200)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,numpy as np
    from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
    from studio.model_cache import stage_cached_tree
    from studio.kaggle_shorts_worker import _safe_extract
    from studio.autonomous_factory import atomic_json
    torch.set_num_threads(min(8,torch.get_num_threads()))
    candidates=[p.parent for p in INPUT.rglob('config.json') if p.parent.name=='alignment'
                and conf['model_kernel_slug'] in p.parts]
    if len(candidates)!=1:raise FileNotFoundError('Validated Russian speech-recognition cache missing')
    home,staged=stage_cached_tree(candidates[0])
    processor=Wav2Vec2Processor.from_pretrained(home,local_files_only=True)
    model=Wav2Vec2ForCTC.from_pretrained(home,local_files_only=True).eval()
    state={'schema':1,'status':'RUNNING','gpu_used':False,'independent_asr':True,
           'orthographic_comparison':'case/punctuation normalized; yo compared to e',
           'stories':{},'listening_review':'pending','published':False}
    started=time.monotonic()
    for sid in range(13,19):
        archives=[p for p in INPUT.rglob(f'TwoWorlds_PRIVATE_{sid}_voice.zip') if conf['voice_kernel_slug'] in p.parts]
        if len(archives)!=1:raise FileNotFoundError('Original full voice recovery archive missing')
        root=OUT/f'voice_inputs/{sid}'
        with zipfile.ZipFile(archives[0]) as z:_safe_extract(z,root)
        project=root/'production_queue'/str(sid);doc=json.loads((project/'project.json').read_text())
        shots={s['scene']:s for s in json.loads((project/'shot_plan.json').read_text())}
        results=[]
        for scene in doc['scenes']:
            with wave.open(str(project/scene['voice']),'rb') as wav:
                rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(np.float32)/32768
            samples=np.interp(np.arange(round(len(pcm)*16000/rate))*rate/16000,np.arange(len(pcm)),pcm).astype(np.float32)
            transcript=[];cursor=0;sr=16000
            while cursor<len(samples):
                if len(samples)-cursor<=24*sr:end=len(samples)
                else:
                    start=cursor+12*sr;stop=cursor+20*sr;step=800
                    energy=[float(np.mean(samples[i:i+step]**2)) for i in range(start,stop,step)]
                    end=start+int(np.argmin(energy))*step+step//2
                value=processor(samples[cursor:end],sampling_rate=sr,return_tensors='pt').input_values
                with torch.inference_mode():ids=model(value).logits.argmax(-1)
                transcript.append(processor.batch_decode(ids)[0]);cursor=end
            recognized=' '.join(transcript)
            expected=shots[scene['id']]['exact_source_text'];reference=words(expected);observed=words(recognized)
            errors=distance(reference,observed);wer=errors/max(len(reference),1)
            results.append({'scene':scene['id'],'reference':expected,'recognized':recognized,
                            'reference_words':len(reference),'recognized_words':len(observed),
                            'word_error_rate':round(wer,4),'requires_review':wer>.18,
                            'source_text_changed':False})
            state['stories'][str(sid)]=results;state['elapsed_s']=round(time.monotonic()-started,2)
            atomic_json(OUT/'batch_voice_qa.json',state)
        print('Completed independent CPU speech audit for story',sid,flush=True)
    state['status']='COMPLETE';state['requires_review']=[{'story':sid,'scene':s['scene'],'wer':s['word_error_rate']}
        for sid,items in state['stories'].items() for s in items if s['requires_review']]
    atomic_json(OUT/'batch_voice_qa.json',state)


if __name__=='__main__':main()
