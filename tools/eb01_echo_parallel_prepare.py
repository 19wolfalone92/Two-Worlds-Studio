"""One real T5 encoding process, exits before either independent video worker."""
from __future__ import annotations
import argparse,gc,json,sys
from pathlib import Path


def main(config):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    from studio.eb01_anatomy_profile import select_specs
    from studio.eb01_v7_storyboard import SHOTS
    from tools.eb01_echo_native_infer import download_locked,sha
    cfg=json.loads(Path(config).read_text());root=Path(cfg['work']);runtime=Path(cfg['runtime'])
    if torch.cuda.device_count()!=2:raise RuntimeError('Two separate T4 required for the bounded benchmark')
    hashes=download_locked(cfg['model_lock'],root/'models')
    (root/'PUBLIC_MODEL_HASHES_SAFE.json').write_text(json.dumps(hashes,sort_keys=True))
    sys.path.insert(0,str(runtime))
    from src.wan_text_encoder import WanT5EncoderModel
    conf=OmegaConf.load(runtime/'config/config.yaml');base=root/'models/base'
    encoder=WanT5EncoderModel.from_pretrained(str(base/conf.text_encoder_kwargs.text_encoder_subpath),
        additional_kwargs=OmegaConf.to_container(conf.text_encoder_kwargs),
        low_cpu_mem_usage=True,torch_dtype=torch.float16).half().eval().to('cuda:1')
    tokenizer=AutoTokenizer.from_pretrained(str(base/conf.text_encoder_kwargs.tokenizer_subpath),local_files_only=True)
    contexts=root/'text-contexts';contexts.mkdir(exist_ok=True)
    seen = set()
    for spec in select_specs(cfg):
        key = spec.get('context_key', spec['shot'])
        if (key, spec['prompt_sha256']) in seen:
            continue
        seen.add((key, spec['prompt_sha256']))
        prompt=spec.get('prompt') or next(s[-1] for s in SHOTS if s[0]==spec['shot'])
        inputs=tokenizer(prompt,padding='max_length',max_length=512,truncation=True,
                         add_special_tokens=True,return_tensors='pt')
        length=int(inputs.attention_mask.sum())
        with torch.inference_mode():
            embeddings=encoder(inputs.input_ids.to('cuda:1'),attention_mask=inputs.attention_mask.to('cuda:1'))[0][:,:length]
        if not torch.isfinite(embeddings).all():raise ValueError('Nonfinite pretrained text context')
        path=contexts/(key+'.safetensors')
        save_file({'prompt_embeds':embeddings.detach().cpu().contiguous()},str(path))
        (contexts/(key+'.json')).write_text(json.dumps({'schema':1,
            'prompt_sha256':spec['prompt_sha256'],'embeddings_sha256':sha(path),
            'model_revisions':{k:v['revision'] for k,v in cfg['model_lock'].items()},
            'source_revision':cfg['studio_revision'],'t5_device':1,'process_exits_before_video_workers':True}))
    print('EB01_SHARED_TEXT_CONTEXTS_READY_RELEASE_BOTH_CARDS',flush=True)
    # Process exit releases T5 and its CUDA context. No live encoder is held
    # by either video worker or counted as free shared VRAM.

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
