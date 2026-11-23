"""Source-locked audio-driven actor experiment, not a production approval."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path
from studio.eb01_v7_storyboard import SHOTS, SRC_SHA, VOICE_SHA, VOICE_SECONDS

BRANCH='feature/eb01-audio-driven-actors-20261120'
UPSTREAM='https://github.com/antgroup/echomimic_v3.git'
UPSTREAM_SHA='7e89489ca51c0d008fc1963ec6c03fc5bd0b9397'
PROFILE='echomimic-v3-flash-dual-t4-fp16-fp32-vae-v1'
PILOTS={'owner':{'shot':'S06','frames':89},'collaborator':{'shot':'S01','frames':65}}
REPOSITORIES={'base':'alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP',
              'flash':'BadToBest/EchoMimicV3','audio':'TencentGameMate/chinese-wav2vec2-base'}


def pilot_spec(alias):
    if alias not in PILOTS:raise ValueError('Unknown independent account')
    item=PILOTS[alias];shot=item['shot'];scene=next(s for s in SHOTS if s[0]==shot)
    return {'profile':PROFILE,'shot':shot,'actor':scene[1],'frames':item['frames'],
            'width':576,'height':1024,'fps':25,'steps':8,'seed':2026112043,
            'portrait_sha256':SRC_SHA[scene[1]],'voice_sha256':VOICE_SHA[shot],
            'voice_seconds':VOICE_SECONDS[shot],'prompt_sha256':hashlib.sha256(scene[-1].encode()).hexdigest(),
            'audio_is_model_input':True,'whole_movie_approved':False,'paid_services':False}


def full_voice_frames(duration,fps=25,tail_seconds=.20):
    import math
    if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0:
        raise ValueError('Invalid original voice clock')
    required=math.ceil((duration+tail_seconds)*fps)
    return 4*math.ceil((required-1)/4)+1


def replace_pinned(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Pinned Echo source boundary changed')
    return text.replace(old, new)


def patch_upstream(runtime:Path):
    """Use two separate devices: T5 on GPU 1, staged video modules on GPU 0.

    Keep upstream neural forward/model weights. Correct T4 SDPA padding,
    honor q/softmax scaling, and avoid advertising FlashAttention on SM75.
    """
    # Both the Flash transformer and the CLIP dependency import this attention.
    for filename in ('wan_transformer3d_audio.py','wan_transformer3d_audio_2512.py'):
        path=runtime/'src'/filename;text=path.read_text()
        tree=ast.parse(text)
        names={'attention','audio_attention','audio_mask_attention'}
        replacements=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                start=sum(len(x) for x in text.splitlines(keepends=True)[:node.lineno-1])
                end=sum(len(x) for x in text.splitlines(keepends=True)[:node.end_lineno])
                signature=ast.get_source_segment(text,node).split('):',1)[0]+'):'
                body='''
    # Exact SDPA equivalent for our unwindowed, deterministic inference path.
    if tuple(window_size) != (-1,-1):
        raise ValueError("Unreviewed sliding-window attention requested")
    batch,ql,heads,_=q.shape
    kl=k.shape[1]
    if q_scale is not None:q=q*q_scale
    mask=None
    if k_lens is not None:
        lengths=torch.as_tensor(k_lens,device=q.device)
        mask=(torch.arange(kl,device=q.device)[None,:]<lengths[:,None])[:,None,None,:]
    output=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        attn_mask=mask,dropout_p=dropout_p,is_causal=causal,scale=softmax_scale).transpose(1,2).contiguous()
    if q_lens is not None:
        lengths=torch.as_tensor(q_lens,device=q.device)
        output=output*(torch.arange(ql,device=q.device)[None,:]<lengths[:,None])[:,:,None,None]
    return output
'''
                replacements.append((start,end,signature+body))
        if not any(ast.parse(new).body[0].name=='attention' for _,_,new in replacements):
            raise ValueError('Pinned upstream attention function missing')
        for start,end,new in sorted(replacements,reverse=True):text=text[:start]+new+text[end:]
        ast.parse(text);path.write_text(text)
    path=runtime/'src/pipeline_wan_fun_inpaint_audio_2512.py';text=path.read_text()
    text=replace_pinned(text,'        device = self._execution_device\n        weight_dtype = self.text_encoder.dtype',
                   '        device = torch.device("cuda:0")\n        weight_dtype = self.text_encoder.dtype')
    text=replace_pinned(text,'prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]',
                   'prompt_embeds = self.text_encoder(text_input_ids.to(self.text_encoder.device), attention_mask=prompt_attention_mask.to(self.text_encoder.device))[0]')
    text=replace_pinned(text,'        # Prepare mask latent variables',
                   '        self.vae.to(device)\n        print("EB01_STAGE_BEGIN_CONDITIONING_VAE",flush=True)\n        # Prepare mask latent variables')
    text=replace_pinned(text,'        # Prepare clip latent variables',
                   '        self.vae.cpu()\n        torch.cuda.empty_cache()\n        self.clip_image_encoder.to(device)\n        # Prepare clip latent variables')
    marker='        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline'
    text=replace_pinned(text,marker,'        self.clip_image_encoder.cpu()\n        torch.cuda.empty_cache()\n        self.transformer.to(device)\n        print("EB01_STAGE_BEGIN_TRANSFORMER",flush=True)\n'+marker)
    text=replace_pinned(text,'            for i, t in enumerate(timesteps):','            for i, t in enumerate(timesteps):\n                print("EB01_STAGE_DENOISE_STEP_START",i+1,flush=True)')
    marker='        if output_type == "numpy":\n'
    text=replace_pinned(text,marker,'''        self.transformer.cpu()
        torch.cuda.empty_cache()
        from safetensors.torch import save_file
        import os
        save_file({"latents":latents.detach().cpu().contiguous()},
                  os.environ["EB01_ECHO_LATENT_CHECKPOINT"])
        self.vae.to(device)
        print("EB01_STAGE_BEGIN_DECODE",flush=True)
'''+marker)
    ast.parse(text);path.write_text(text)
    return {str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        runtime/'src/wan_transformer3d_audio.py',runtime/'src/wan_transformer3d_audio_2512.py',path)}
