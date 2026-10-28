"""Deterministic, original procedural comic panels: offline placeholder art, not AI image generation.

Comic compositions establish geography, wardrobe and scene blocking. An illustrator can
replace them by specifying a rights-cleared 'background' image in episode JSON.
"""
from __future__ import annotations

import math
from PIL import Image, ImageDraw, ImageFont

W, H = 540, 960
INK = (11, 13, 28)
PAPER = (225, 215, 206)
PALETTES = {
    "earth": ((8, 19, 41), (28, 67, 98), (57, 105, 143)),
    "noctaris": ((22, 11, 35), (80, 31, 61), (124, 54, 81)),
}


def box(draw, bounds, fill, outline=INK, width=5):
    draw.rounded_rectangle(bounds, radius=8, fill=fill, outline=outline, width=width)


def character(draw, x, y, scale=1.0, armor=False, side=False, hair="dark"):
    """Draw a reproducible stylized person (head centered at x,y)."""
    s = scale
    def xy(a, b):
        return (round(x+a*s), round(y+b*s))
    dark_hair = (25, 19, 36) if hair == "dark" else (70, 33, 49)
    skin = (214, 155, 141)
    # silhouette / coat
    draw.polygon([xy(-52,64),xy(54,64),xy(93,239),xy(-85,239)],
                 fill=(29, 39, 59) if not armor else (45, 44, 70))
    draw.line([xy(-52,64), xy(-85,239), xy(93,239),xy(54,64)], fill=INK,width=max(2,round(5*s)),joint="curve")
    if armor:
        draw.polygon([xy(-56,70),xy(0,101),xy(56,70),xy(62,166),xy(0,196),xy(-65,166)],
                     fill=(84, 82, 113),outline=(170, 171, 196),width=max(2,round(4*s)))
        draw.line([xy(-60,89),xy(0,117),xy(60,89)],fill=(208,189,169),width=max(2,round(5*s)))
        draw.polygon([xy(-20,99),xy(0,118),xy(20,99),xy(0,146)],fill=(170,104,130))
    else:
        draw.line([xy(0,97),xy(0,213)], fill=(116,125,147),width=max(2,round(6*s)))
        draw.line([xy(-46,76),xy(-16,137),xy(0,98),xy(17,137),xy(49,78)],fill=(178,180,188),width=max(2,round(3*s)))
    # hair behind and neck
    draw.ellipse([xy(-45,-68),xy(47,78)], fill=dark_hair, outline=INK,width=max(2,round(4*s)))
    draw.rectangle([xy(-15,40),xy(15,87)],fill=skin)
    draw.ellipse([xy(-43,-59),xy(44,57)],fill=skin,outline=INK,width=max(2,round(4*s)))
    # face
    if side:
        draw.arc([xy(-22,-19),xy(20,6)],190,330,fill=INK,width=max(2,round(3*s)))
        draw.line([xy(6,20),xy(18,23)], fill=(117,73,76), width=max(2,round(3*s)))
    else:
        for eye_x in (-19,19):
            draw.ellipse([xy(eye_x-7,-6),xy(eye_x+7,2)],fill=(239,238,228),outline=INK,width=max(1,round(2*s)))
            draw.ellipse([xy(eye_x-3,-5),xy(eye_x+3,2)],fill=(57,101,104))
        draw.line([xy(-4,3),xy(-7,25),xy(3,27)],fill=(128,81,83),width=max(2,round(2*s)))
        draw.arc([xy(-12,23),xy(13,39)],5,170,fill=(112,64,78),width=max(2,round(3*s)))
    # distinctive asymmetrical hairstyle and Lyra's scar
    draw.polygon([xy(-47,-34),xy(-29,-77),xy(33,-74),xy(48,-34),xy(18,-46),xy(2,-27),xy(-14,-50)],
                 fill=dark_hair, outline=INK)
    if armor:
        draw.line([xy(22,3),xy(31,25)],fill=(145,69,79),width=max(2,round(2*s)))
    # highlights
    draw.arc([xy(-44,-64),xy(35,53)],180,255,fill=(91,83,111),width=max(2,round(3*s)))


def earth_city(d, daytime=False):
    sky = (62,100,141) if not daytime else (243,156,117)
    d.rectangle((0,0,W,H),fill=sky)
    for y in range(0,600,16):
        a=y/600
        d.rectangle((0,y,W,y+16),fill=tuple(round(sky[i]*(1-a)+((17,25,57) if not daytime else (255,211,174))[i]*a) for i in range(3)))
    if daytime:
        d.ellipse((344,85,470,211),fill=(255,238,179),outline=(250,197,140),width=7)
    else:
        d.ellipse((350,72,441,163),fill=(216,232,235),outline=(140,189,211),width=6)
    for i in range(18):
        x=i*34-15
        top=260+((i*47)%190)
        d.rectangle((x,top,x+40,720),fill=(19+i%4*7,31+i%5*5,53+i%3*9),outline=INK,width=3)
        for wy in range(top+17,650,38):
            for wx in (x+8,x+24):
                if (i+wy//38+wx//8)%4!=0:
                    d.rectangle((wx,wy,wx+7,wy+14),fill=(240,180,111) if daytime else (115,174,203))
    d.polygon([(0,705),(540,675),(540,960),(0,960)],fill=(15,24,39))
    for i in range(14):
        x=(i*71)%W
        d.line((x,180,x-12,580),fill=(161,208,232,80),width=2)


def nox_city(d):
    d.rectangle((0,0,W,H),fill=(63,23,59))
    for i in range(3):
        yy=i*230
        d.rectangle((0,yy,W,yy+230),fill=((55+i*13),24+i*8,57+i*6))
    d.ellipse((352,82,466,196),fill=(200,208,220),outline=(107,119,157),width=7)
    d.ellipse((65,115,151,201),fill=(210,141,174),outline=(144,91,120),width=5)
    for i in range(11):
        x=i*62-26
        top=350+((i*83)%170)
        d.rectangle((x,top,x+66,740),fill=(27,18,40),outline=INK,width=4)
        d.polygon([(x-8,top),(x+33,top-95),(x+75,top)],fill=(36,24,54),outline=INK)
        for wy in range(top+40,710,65):
            d.polygon([(x+19,wy+14),(x+19,wy),(x+29,wy-12),(x+40,wy),(x+40,wy+14)],
                      fill=(222,108,130))
    d.polygon([(0,700),(130,665),(260,718),(385,651),(540,696),(540,960),(0,960)],fill=(23,12,31))


def mirror(d, x=278,y=281,rw=170,rh=240, portal=False):
    bounds=(x-rw//2,y-rh//2,x+rw//2,y+rh//2)
    d.ellipse((bounds[0]-20,bounds[1]-23,bounds[2]+20,bounds[3]+23),
              fill=(137,111,88),outline=INK,width=7)
    d.ellipse((bounds[0]-8,bounds[1]-10,bounds[2]+8,bounds[3]+11),
              fill=(54,96,107) if not portal else (104,44,99),outline=(222,185,132),width=7)
    for a in range(0,360,36):
        rx=x+round(math.cos(math.radians(a))*(rw/2+14))
        ry=y+round(math.sin(math.radians(a))*(rh/2+17))
        d.ellipse((rx-5,ry-5,rx+5,ry+5),fill=(232,199,140),outline=INK,width=2)
    d.polygon([(x-18,y+rh//2+19),(x+18,y+rh//2+19),(x+55,y+rh//2+78),(x-55,y+rh//2+78)],
              fill=(109,87,84),outline=INK)


def dramatic_grid(d):
    # inked edges and graphic-novel texture
    for x in (0,540):
        d.line((x,0,x,H),fill=INK,width=12)
    d.line((0,0,540,0),fill=INK,width=10)
    for i in range(15):
        x=(i*131+57)%W
        y=(i*193+71)%600
        d.line((x,y,x+25,y+7),fill=(241,214,205),width=2)


def panel(scene: dict, width=540, height=960) -> Image.Image:
    """Scene-specific free illustrated composition for the E01 shot vocabulary."""
    kind=scene.get("art","")
    result=Image.new("RGB",(W,H),(10,19,39))
    d=ImageDraw.Draw(result)
    if kind in ("earth_photo","earth_mirror","sunrise"):
        earth_city(d,daytime=kind=="sunrise")
    elif kind in ("two_worlds","exchange"):
        earth_city(d)
        d.polygon([(W//2,0),(W,0),(W,960),(W//2,960)],fill=(70,29,58))
        moon=[(365,125,460,220),(300,150,370,220)]
        for i,b in enumerate(moon):
            d.ellipse(b,fill=(211,188,207) if i else (211,213,231))
    else:
        nox_city(d)
    if kind=="earth_photo":
        box(d,(33,98,490,609),(17,29,50),width=9)
        earth_city_window=(73,130,452,512)
        box(d,earth_city_window,(37,70,103),outline=(168,182,184),width=6)
        d.line((260,130,260,512),fill=(167,189,190),width=8)
        d.line((73,322,452,322),fill=(167,189,190),width=8)
        for i in range(29):
            rx=80+(i*113)%360
            ry=145+(i*73)%345
            d.line((rx,ry,rx-11,ry+27),fill=(126,194,223),width=3)
        character(d,171,435,1.15)
        box(d,(240,535,377,632),(123,101,105),outline=(224,189,130),width=7)
        d.ellipse((277,552,333,607),fill=(209,155,141),outline=INK,width=3)
        d.line((306,555,306,598),fill=(43,28,42),width=6)
    elif kind=="earth_mirror":
        d.polygon([(0,635),(540,581),(540,960),(0,960)],fill=(20,29,49))
        mirror(d,355,310,217,318)
        character(d,350,304,.70,armor=True)
        character(d,119,382,1.32,armor=False,side=True)
        d.line((190,370,240,389),fill=(225,174,150),width=7)
    elif kind=="two_worlds":
        d.line((270,0,270,960),fill=(236,196,162),width=11)
        character(d,138,335,1.23)
        character(d,400,335,1.23,armor=True)
        d.ellipse((222,270,318,370),outline=(218,189,211),width=11)
    elif kind=="exchange":
        mirror(d,270,296,310,430,portal=True)
        character(d,111,376,.95)
        character(d,432,376,.95,armor=True)
        for r in (70,115,160,205):
            d.arc((270-r,292-r,270+r,292+r),30,153,fill=(255,195,217),width=5)
        for i in range(30):
            x=(i*71+25)%500+15
            y=(i*47+58)%620
            d.line((x,y,x+18,y-26),fill=(250,218,198),width=4)
    elif kind=="fortress":
        box(d,(55,60,475,587),(29,21,47),outline=(139,113,142),width=12)
        d.ellipse((260,98,340,176),fill=(219,212,227))
        d.ellipse((356,115,411,170),fill=(215,147,178))
        for x in (65,440):
            d.rectangle((x,95,x+35,665),fill=(54,39,70),outline=INK,width=5)
        character(d,224,360,1.35,armor=True)
        character(d,418,466,.85,armor=True,side=True)
        d.line((416,543,352,590),fill=(219,190,155),width=8)
    elif kind=="sunrise":
        box(d,(36,55,490,597),(20,35,63),outline=(226,210,181),width=10)
        d.rectangle((65,96,466,530),fill=(239,153,123))
        d.ellipse((317,155,444,280),fill=(255,228,172))
        for x in (118,218,390):
            d.rectangle((x,370,x+75,530),fill=(62,61,84))
        d.line((266,96,266,530),fill=(213,219,210),width=10)
        character(d,192,365,1.24,side=True)
    else:
        # Unrecognized art type stays a simple generic illustration rather than erroring.
        character(d,250,390,1.25,armor=scene.get("palette")=="noctaris")
    # Inking frames and a dramatic diagonal corner label, without the episode title.
    dramatic_grid(d)
    d.polygon([(0,0),(188,0),(0,155)],fill=(24,27,45),outline=INK)
    d.line((8,167,193,9),fill=(233,166,151),width=5)
    return result.resize((width,height),Image.Resampling.LANCZOS)
