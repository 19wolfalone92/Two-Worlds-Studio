"""TWO WORLDS Studio: conservative single-illustration puppet preparation.

Input is an already alpha-matted FULL figure in ONE approved camera view.
Template landmarks are estimates, not AI pose detection: exported parts remain
UNAPPROVED until anatomy, occlusion and hidden-joint repaint are reviewed.
The output is compatible with studio.character_engine.CharacterRig.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from studio.character_engine import CharacterRig

# Zones describe a single upright silhouette, relative to its visible box.
ZONES = {
    "head": [( .26,0),(.74,0),(.78,.23),(.22,.23)],
    "upper_arm": [(.21,.22),(.37,.23),(.36,.46),(.16,.46)],
    "forearm": [(.16,.44),(.36,.44),(.34,.60),(.12,.63)],
    "hand": [(.12,.60),(.34,.59),(.33,.71),(.11,.71)],
    "far_upper_arm": [(.63,.23),(.79,.22),(.84,.46),(.64,.46)],
    "far_forearm": [(.64,.44),(.84,.44),(.88,.63),(.66,.60)],
    "far_hand": [(.66,.59),(.88,.60),(.89,.71),(.67,.71)],
    "left_leg": [(.31,.56),(.51,.56),(.52,.80),(.28,.80)],
    "left_shin": [(.28,.78),(.52,.78),(.51,.965),(.28,.965)],
    "left_foot": [(.22,.95),(.52,.95),(.52,1),(.20,1)],
    "right_leg": [(.51,.56),(.70,.56),(.72,.80),(.49,.80)],
    "right_shin": [(.49,.78),(.72,.78),(.74,.965),(.49,.965)],
    "right_foot": [(.49,.95),(.79,.95),(.81,1),(.49,1)],
}
PIVOTS = {
    "torso":(.5,.47), "head":(.5,.20),
    "upper_arm":(.28,.27),"forearm":(.25,.44),"hand":(.24,.61),
    "far_upper_arm":(.72,.27),"far_forearm":(.75,.44),"far_hand":(.76,.61),
    "left_leg":(.40,.58),"left_shin":(.40,.78),"left_foot":(.40,.96),
    "right_leg":(.60,.58),"right_shin":(.60,.78),"right_foot":(.60,.96),
}
PARENTS = {"head":"torso", "upper_arm":"torso", "forearm":"upper_arm",
           "hand":"forearm", "far_upper_arm":"torso",
           "far_forearm":"far_upper_arm", "far_hand":"far_forearm",
           "left_leg":"torso", "left_shin":"left_leg", "left_foot":"left_shin",
           "right_leg":"torso", "right_shin":"right_leg", "right_foot":"right_shin"}
Z_ORDER = {"far_upper_arm":1,"far_forearm":2,"far_hand":3,"left_leg":4,
           "left_shin":5,"left_foot":6,"right_leg":7,"right_shin":8,
           "right_foot":9,"torso":10,"head":11,"upper_arm":12,"forearm":13,"hand":14}


def two_bone_ik(hip, foot, upper:float, lower:float, bend:int=1):
    """Knee and joint rotations in screen degrees for reachable foot targets."""
    ax,ay=map(float,hip); bx,by=map(float,foot)
    upper=float(upper);lower=float(lower)
    dist=math.hypot(bx-ax,by-ay)
    if (not all(math.isfinite(x) for x in (ax,ay,bx,by,upper,lower))
        or min(upper,lower)<=0 or bend not in (-1,1)
        or not abs(upper-lower)+1e-6<dist<upper+lower-1e-6):
        raise ValueError('IK target unreachable or parameters invalid')
    along=(upper*upper-lower*lower+dist*dist)/(2*dist)
    height=math.sqrt(max(0,upper*upper-along*along))
    ux,uy=(bx-ax)/dist,(by-ay)/dist
    knee=(ax+ux*along-bend*uy*height,ay+uy*along+bend*ux*height)
    upper_angle=math.degrees(math.atan2(-(knee[0]-ax),knee[1]-ay))
    lower_angle=math.degrees(math.atan2(-(bx-knee[0]),by-knee[1]))
    return knee, upper_angle, (lower_angle-upper_angle+180)%360-180


def walk_tracks(box:tuple[int,int,int,int],canvas:tuple[int,int]) -> dict:
    """Two-phase keyframed walk with planted-world foot targets during stance.

    Swing feet lift; artwork still needs a visual foot-skating QA and retouch.
    """
    x0,y0,x1,y1=box; w,h=canvas; bw=x1-x0;bh=y1-y0
    xs=[0,.125,.25,.375,.50,.625,.75,.875,1]
    travel=.065*bw
    upper=.205*bh;lower=.205*bh
    hip_y=y0+.575*bh;floor=y0+.962*bh
    frames={name:[] for name in ('__root__','head','upper_arm','forearm','far_upper_arm',
                                 'left_leg','left_shin','right_leg','right_shin')}
    for t in xs:
        dx=(t-.5)*travel
        frames['__root__'].append({'at':t,'x':dx/w,'y':.002*math.sin(t*4*math.pi)*bh/h})
        frames['head'].append({'at':t,'angle':1.2*math.sin(t*2*math.pi)})
        frames['upper_arm'].append({'at':t,'angle':-24*math.sin(t*math.pi/2)})
        frames['forearm'].append({'at':t,'angle':-35*math.sin(t*math.pi/2)})
        frames['far_upper_arm'].append({'at':t,'angle':4*math.sin(t*4*math.pi)})
        for side,off,bend in (('left',-.10,-1),('right',.10,1)):
            base_x=x0+(0.5+off)*bw
            # Left plants in the first half, right plants in the second.
            if side=='left':
                swinging=t>.5
                u=(t-.5)*2 if swinging else 0.0
            else:
                swinging=t<.5
                u=t*2 if swinging else 1.0
            hip_x=base_x+dx
            foot_x=base_x+travel*u
            foot_y=floor-.022*bh*4*u*(1-u) if swinging else floor
            _,a,b=two_bone_ik((hip_x,hip_y),(foot_x,foot_y),upper,lower,bend)
            frames[f'{side}_leg'].append({'at':t,'angle':round(a,4)})
            frames[f'{side}_shin'].append({'at':t,'angle':round(b,4)})
    return frames


def _scene_pivot(box,canvas,coord):
    x0,y0,x1,y1=box;w,h=canvas
    return [(x0+(x1-x0)*coord[0])/w,(y0+(y1-y0)*coord[1])/h]


def prepare_auto_rig(portrait:Path,out:Path, *,body_id:str,
                     camera_view:str='back', facing:str='mirror')->dict:
    """Create PNG sprites and playable rig, never auto-approve the art."""
    if camera_view not in {'back','front','profile','three_quarter','rear_three_quarter'}:
        raise ValueError('Unsupported explicit camera view')
    if not body_id or not body_id.replace('_','').isalnum():
        raise ValueError('Invalid body identifier')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Do not overwrite a previous rig or approved art')
    with Image.open(portrait) as image:
        image.load()
        if image.format!='PNG' or 'A' not in image.getbands():
            raise ValueError('Auto-rig needs a genuine transparent PNG, not a matte JPEG')
        rgba=image.convert('RGBA')
    w,h=rgba.size
    if w<96 or h<160 or w*h>12_000_000:
        raise ValueError('Unsupported rig source dimensions')
    alpha=np.asarray(rgba.getchannel('A'),dtype=np.uint8)
    if float((alpha[0,:]>8).mean()+(alpha[-1,:]>8).mean()+
             (alpha[:,0]>8).mean()+(alpha[:,-1]>8).mean())>.08:
        raise ValueError('Character touches canvas edge; no reliable silhouette')
    box=rgba.getchannel('A').point(lambda a:255 if a>32 else 0).getbbox()
    if box is None:
        raise ValueError('No visible character')
    x0,y0,x1,y1=box; bw=x1-x0;bh=y1-y0
    if not .18<=bw/bh<=.85 or bh/h<.60:
        raise ValueError('Figure is not a complete upright standing actor')
    out.mkdir(parents=True,exist_ok=True)
    arrays=np.asarray(rgba,dtype=np.uint8)
    remaining=np.ones((h,w),dtype=np.uint8)*255
    parts=[]
    # Distal pieces first; every original visible pixel belongs to one part.
    for name,polygon in ZONES.items():
        mask=Image.new('L',(w,h),0)
        dr=ImageDraw.Draw(mask)
        dr.polygon([(round(x0+x*bw),round(y0+y*bh)) for x,y in polygon],fill=255)
        m=np.asarray(mask,dtype=np.uint8)
        m=np.minimum(m,remaining)
        remaining[m>0]=0
        a=arrays.copy();a[:,:,3]=np.minimum(a[:,:,3],m)
        if a[:,:,3].max()<16:
            raise ValueError(f'Empty inferred limb: {name}. Provide new clean full-body art')
        target=out/f'{name}.png';Image.fromarray(a,'RGBA').save(target)
        parts.append({'name':name,'parent':PARENTS.get(name),'pivot':_scene_pivot(box,(w,h),PIVOTS[name]),
                      'z':Z_ORDER[name],'path':target.name,
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'review_status':'unreviewed','rights':'original',
                      'rights_note':'Automatically segmented from pinned source; artist repair required.'})
    torso=arrays.copy();torso[:,:,3]=np.minimum(torso[:,:,3],remaining)
    if torso[:,:,3].max()<16:
        raise ValueError('Torso vanished during silhouette segmentation')
    target=out/'torso.png';Image.fromarray(torso,'RGBA').save(target)
    parts.append({'name':'torso','pivot':_scene_pivot(box,(w,h),PIVOTS['torso']),
                  'z':Z_ORDER['torso'],'path':target.name,
                  'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                  'review_status':'unreviewed','rights':'original',
                  'rights_note':'Residual torso pixels; hidden joints need manual overpainting.'})
    config={'schema':1,'body_id':body_id,'camera_view':camera_view,'facing':facing,
            'root_pivot':_scene_pivot(box,(w,h),(.5,.89)),
            'parts':parts,'tracks':walk_tracks(box,(w,h)),
            'ground_shadow':{'center':_scene_pivot(box,(w,h),(.5,.967)),
                            'radii':[.16*bw/w,.016*bh/h], 'opacity':.25,'blur':.01},
            'source_image_sha256':hashlib.sha256(Path(portrait).read_bytes()).hexdigest(),
            'segmentation_method':'geometric_template_unreviewed',
            'autorig_anatomy_verified':False,'owner_approved':False,
            'publication_approved':False,
            'known_limitations':['Requires artist repaint for hidden joints',
                                 'Silhouette template not a trained pose estimator',
                                 'Gait is keyframed 2D; visual foot-contact QC required']}
    (out/'rig.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    CharacterRig(config,out,(w,h))
    return config


def load_auto_rig(folder:Path)->CharacterRig:
    cfg=json.loads((folder/'rig.json').read_text(encoding='utf-8'))
    if cfg.get('owner_approved') or cfg.get('publication_approved'):
        raise ValueError('Never forge approvals by editing an autogenerated manifest')
    return CharacterRig(cfg,folder,Image.open(folder/'torso.png').size)


def main():
    """Prepare the unapproved rig for private owner review; no publication."""
    import argparse
    parser=argparse.ArgumentParser(description='TWO WORLDS private animator rig preparation')
    parser.add_argument('--image',type=Path,required=True,
                        help='Existing private full-figure transparent PNG (correct camera angle)')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--body',required=True)
    parser.add_argument('--view',default='back',
                        choices=('back','front','profile','three_quarter','rear_three_quarter'))
    parser.add_argument('--facing',default='mirror')
    opts=parser.parse_args()
    result=prepare_auto_rig(opts.image,opts.output,body_id=opts.body,
                            camera_view=opts.view,facing=opts.facing)
    print(json.dumps({'rig':str(opts.output/'rig.json'),
                      'parts':len(result['parts']),
                      'review':'ARTIST_REPAIR_REQUIRED',
                      'approved':False,
                      'publication_allowed':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
