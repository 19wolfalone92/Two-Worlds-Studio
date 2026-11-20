"""Bounded single-card workers after shared text encoding, no production launch."""
import hashlib
from pathlib import Path
from studio.eb01_echo_profile import UPSTREAM_SHA
from studio.eb01_v7_storyboard import SHOTS,SRC_SHA,VOICE_SHA,VOICE_SECONDS
PROFILE='echomimic-v3-flash-precomputed-text-single-t4-v2'


def specs(account):
    if account not in ('owner','collaborator'):raise ValueError('Unknown independent account')
    definitions=([('S06',[45,45]),('S03',[41])] if account=='owner' else [('S01',[33,33]),('S03',[41])])
    result=[]
    for slot,(shot,segments) in enumerate(definitions):
        scene=next(s for s in SHOTS if s[0]==shot)
        result.append({'profile':PROFILE,'account_alias':account,'physical_gpu_index':slot,'shot':shot,
            'actor':scene[1],'segment_frames':segments,'frames':sum(segments)-len(segments)+1,
            'width':576,'height':1024,'fps':25,'steps':8,
            'seed':2026112150+(100 if account=='collaborator' else 0)+slot*10,
            'portrait_sha256':SRC_SHA[scene[1]],'voice_sha256':VOICE_SHA[shot],
            'voice_seconds':VOICE_SECONDS[shot],
            'prompt_sha256':hashlib.sha256(scene[-1].encode()).hexdigest(),
            'audio_is_model_input':True,'benchmark_only':True,'whole_movie_approved':False,
            'human_artistic_approved':False})
    return result


def upgrade_precomputed(runtime:Path):
    """After the guarded original patch, use provided embeddings and vision only."""
    p=runtime/'src/pipeline_wan_fun_inpaint_audio_2512.py';text=p.read_text()
    old='        weight_dtype = self.text_encoder.dtype'
    if text.count(old)!=1:raise ValueError('Pinned precomputed dtype boundary changed')
    text=text.replace(old,'        weight_dtype = torch.float16')
    old='        self.clip_image_encoder.to(device)'
    if text.count(old)!=1:raise ValueError('Pinned image encoder staging boundary changed')
    text=text.replace(old,'        self.clip_image_encoder.model.visual.to(device)')
    p.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
