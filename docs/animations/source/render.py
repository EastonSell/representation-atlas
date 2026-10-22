#!/usr/bin/env python3
"""Deterministic Pillow / ffmpeg teaching animations. No installs or external assets.

From docs/animations:
python -B source/render.py --scene coordinate-readout --out .
python -B -m unittest discover -s source -v
"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
import concepts as c

W,H=1280,720
BG='#0c1422'; PANEL='#142238'; INK='#f0f4fc'; MUTED='#acbad0'; GRID='#2c405b'
CYAN='#54d8df'; AMBER='#ffd285'; CORAL='#ff9b99'; GREEN='#a6e7be'
REGULAR='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
MONO='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
LAYOUT_FAILURES=[]


@lru_cache(None)
def font(size=26,bold=False,mono=False):
    return ImageFont.truetype(MONO if mono else BOLD if bold else REGULAR,size)


def text(d,xy,value,size=26,color=INK,bold=False,anchor=None,mono=False):
    bounds=d.textbbox(xy,str(value),font=font(size,bold,mono),anchor=anchor)
    if bounds[0]<0 or bounds[2]>W or bounds[1]<0 or bounds[3]>H:
        LAYOUT_FAILURES.append({'text':str(value),'bounds':bounds})
    d.text(xy,str(value),font=font(size,bold,mono),fill=color,anchor=anchor)


def box(d,rect,fill=PANEL,outline=None,radius=18,width=2):
    d.rounded_rectangle(rect,radius=radius,fill=fill,outline=outline,width=width)


def wrapped(d,value,width,size=26,bold=False):
    words=value.split();lines=[];line=''
    for word in words:
        candidate=(line+' '+word).strip()
        if d.textlength(candidate,font=font(size,bold)) > width and line:
            lines.append(line);line=word
        else:line=candidate
    if line:lines.append(line)
    return lines


def paragraph(d,xy,value,width,size=26,color=INK,bold=False,line_height=None,max_height=None):
    x,y=xy;lh=line_height or int(size*1.35)
    lines=[]
    for part in value.split('\n'):lines.extend(wrapped(d,part,width,size,bold))
    if max_height and len(lines)*lh>max_height:
        LAYOUT_FAILURES.append({'text':value,'height':len(lines)*lh,'available':max_height})
    for i,line in enumerate(lines):text(d,(x,y+i*lh),line,size,color,bold)
    return y+len(lines)*lh


def arrow(d,start,end,color=CYAN,width=4,head=12):
    d.line([start,end],fill=color,width=width)
    dx,dy=end[0]-start[0],end[1]-start[1];mag=math.hypot(dx,dy)
    if mag<1:return
    ux,uy=dx/mag,dy/mag;px,py=-uy,ux
    d.polygon([end,(end[0]-head*ux+head*.5*px,end[1]-head*uy+head*.5*py),
                  (end[0]-head*ux-head*.5*px,end[1]-head*uy-head*.5*py)],fill=color)


def dot(d,xy,color,r=10):d.ellipse((xy[0]-r,xy[1]-r,xy[0]+r,xy[1]+r),fill=color)


def ease(x):x=max(0.,min(1.,x));return x*x*(3-2*x)


def common(scene,t):
    story=c.STORIES[scene]; idx=min(3,int(t//6)); shot=story['scenes'][idx]
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(48,17),story['badge'],18,AMBER,True)
    text(d,(48,52),story['title'],40,INK,True)
    text(d,(48,105),story['subtitle'],24,MUTED)
    for j,item in enumerate(story['scenes']):
        x=48+j*300;active=j==idx
        box(d,(x,145,x+284,177),fill=CYAN if active else PANEL,radius=8)
        nav=f"{j+1}  {item['label']}"; nav_size=18
        while d.textlength(nav,font=font(nav_size,active))>262:nav_size-=1
        text(d,(x+11,149),nav,nav_size,BG if active else MUTED,active)
    box(d,(48,591,1232,684),fill=PANEL,radius=12)
    paragraph(d,(70,606),shot['caption'],1140,26,INK,line_height=34,max_height=70)
    d.rectangle((48,701,1232,704),fill=GRID)
    d.rectangle((48,701,48+1184*min(t/c.DURATION,1),704),fill=CYAN)
    return im,d,idx,t-idx*6


def title(d,label,x=76,y=206,color=CYAN,size=30):text(d,(x,y),label,size,color,True)


def card(d,rect,label,color=CYAN):
    box(d,rect,outline=GRID)
    title(d,label,rect[0]+23,rect[1]+20,color,size=28)


def replay(d,idx,u):
    r=c.replay_results()
    if idx==0:
        for rect,label,lines,col in [((48,203,430,433),'Task A',['input (1, 0)','target 1'],CYAN),
                                   ((448,203,830,433),'Task B',['input (1, 1)','target 0'],CORAL),
                                   ((848,203,1232,433),'Both can be solved',['weights (1, −1)','A → 1    B → 0'],GREEN)]:
            card(d,rect,label,col)
            for j,l in enumerate(lines):text(d,(rect[0]+25,rect[1]+88+j*48),l,29)
        text(d,(70,465),'prediction = w1 × x1 + w2 × x2',31,INK,True)
        text(d,(70,518),'After learning A: weights ≈ (1, 0). What happens when B arrives?',25,MUTED)
    elif idx==1:
        for x,name,label,col in [(48,'current_only','B only',CORAL),(650,'replay','B + stored A',GREEN)]:
            a=r[name]
            update=round(120*ease((u-.8)/3.7)); w=r['after_a']['weights']
            batch=(c.TASK_B,c.TASK_B) if name=='current_only' else (c.TASK_B,c.TASK_A)
            for _ in range(update):w=c.step(w,batch)
            a=dict(a,a_loss=c.loss(w,c.TASK_A),b_loss=c.loss(w,c.TASK_B))
            card(d,(x,203,x+582,550),label,col)
            text(d,(x+25,263),f'weights ≈ ({w[0]:.3f}, {w[1]:.3f})',26)
            text(d,(x+25,315),'A prediction  (target 1)',24,MUTED)
            box(d,(x+25,354,x+475,388),fill=BG,radius=5)
            d.rounded_rectangle((x+25,354,x+25+450*w[0],388),radius=5,fill=col)
            text(d,(x+488,354),f'{w[0]:.3f}',23,col,True)
            text(d,(x+25,415),f"A loss {a['a_loss']:.6f}  |  B loss {a['b_loss']:.6f}",23)
            text(d,(x+25,465),f'Later update {update}/120 · {2*update} presentations',22,MUTED)
            text(d,(x+25,503),f"Stored old examples: {a['stored_old_examples']}",23,MUTED)
    elif idx==2:
        text(d,(72,201),'Virtual B step: (1, 0) → (0.75, −0.25)',27)
        text(d,(72,242),'Same candidate set. Half-squared error.',22,MUTED)
        columns=[(72,'Old example'),(530,'Loss now'),(815,'Signed loss change')]
        for x,label in columns:text(d,(x,290),label,24,MUTED,True)
        for j,row in enumerate(c.selection_scores()):
            y=337+j*63;box(d,(61,y-5,1218,y+49),fill=PANEL,radius=8)
            text(d,(76,y+5),row['name'],26)
            high=j==1;threat=j==2
            if high:box(d,(516,y-1,718,y+45),fill='#41313a',outline=CORAL,radius=7)
            if threat:box(d,(803,y-1,1140,y+45),fill='#203b3e',outline=CYAN,radius=7)
            text(d,(533,y+5),f"{row['now']:.5f}".rstrip('0').rstrip('.'),26,CORAL if high else INK,high)
            text(d,(820,y+5),f"{row['delta']:+.5f}",26,CYAN if threat else INK,threat)
        text(d,(76,546),'High-loss → Hard now     Predicted-interference → Threatened next',23,AMBER)
    else:
        card(d,(48,211,624,459),'1  Buffer retention',CYAN)
        paragraph(d,(76,277),'Which old examples remain available?',510,32,line_height=45)
        text(d,(76,397),'Memory access is a resource.',24,MUTED)
        card(d,(650,211,1232,459),'2  Replay selection',CORAL)
        paragraph(d,(678,277),'Which available examples are rehearsed next?',510,32,line_height=45)
        text(d,(678,397),'A score is not guaranteed protection.',24,MUTED)
        text(d,(76,502),'This toy is not a benchmark comparison or a biological simulation.',26,AMBER)


def axes(d,origin=(200,510),scale=85,xrange=4,yrange=4):
    ox,oy=origin
    for i in range(xrange+1):
        x=ox+i*scale;d.line((x,oy,x,oy-yrange*scale),fill=GRID,width=1)
        text(d,(x,oy+10),str(i),18,MUTED,anchor='mt')
    for j in range(yrange+1):
        y=oy-j*scale;d.line((ox,y,ox+xrange*scale,y),fill=GRID,width=1)
        if j:text(d,(ox-15,y),str(j),18,MUTED,anchor='rm')
    arrow(d,(ox,oy),(ox+xrange*scale+25,oy),MUTED,2)
    arrow(d,(ox,oy),(ox,oy-yrange*scale-15),MUTED,2)
    return lambda p:(ox+p[0]*scale,oy-p[1]*scale)


def geometry(d,idx,u):
    if idx==0:
        box(d,(48,196,708,566),outline=GRID)
        xy=axes(d,(149,510),80,5,3)
        q=ease((u-1.2)/3)
        a=(1+2*q,2*q);b=(2*q,1+2*q)
        aa,bb=xy(a),xy(b)
        arrow(d,xy((0,0)),aa,CYAN,3);arrow(d,xy((0,0)),bb,CORAL,3)
        d.line((aa,bb),fill=AMBER,width=5);dot(d,aa,CYAN);dot(d,bb,CORAL)
        text(d,(aa[0]+15,aa[1]+8),'A',25,CYAN,True)
        text(d,(bb[0]-18,bb[1]-23),'B',25,CORAL,True,anchor='ra')
        text(d,(89,212),f'Shared offset = ({2*q:.1f}, {2*q:.1f})',23)
        card(d,(738,196,1232,566),'One motion, two metrics')
        text(d,(764,280),'Distance',25,MUTED);text(d,(764,317),'√2 ≈ 1.414  (unchanged)',28,AMBER,True)
        text(d,(764,386),'Origin-based cosine',25,MUTED);text(d,(764,423),f'{c.cosine(a,b):.3f}',39,CYAN,True)
        text(d,(764,495),'0 → 12/13 ≈ 0.923',25,MUTED)
    elif idx==1:
        cols=[(48,'Original states','score = x1 − 0.5',('+0.5','−0.5'),CYAN),
              (450,'Translate states','same fixed readout',('+2.5','+1.5'),CORAL),
              (852,'Compensate bias','score = x1 − 2.5',('+0.5','−0.5'),GREEN)]
        for x,label,desc,values,col in cols:
            card(d,(x,205,x+380,485),label,col)
            text(d,(x+22,275),desc,23)
            text(d,(x+22,332),f'A   {values[0]}',31)
            text(d,(x+22,386),f'B   {values[1]}',31,col,True)
            text(d,(x+22,448),'Zero-threshold decision',20,MUTED)
        text(d,(77,524),'B changes sign with the fixed readout. Restoring the bias is another edit.',25,AMBER)
    elif idx==2:
        box(d,(48,199,706,566),outline=GRID)
        origin=(215,382);scale=123
        arrow(d,(85,382),(650,382),MUTED,2);arrow(d,(215,548),(215,249),MUTED,2)
        text(d,(628,397),'x',22,MUTED);text(d,(187,249),'y',22,MUTED)
        q=ease((u-1.0)/3.0)
        for sign,col,label in [(1,CYAN,'+y label'),(-1,CORAL,'−y label')]:
            start=(338,382-sign*123);end=(338,382-sign*123*(1-q))
            d.line((start,(338,382)),fill=GRID,width=2)
            dot(d,end,col,r=10)
            if q>.95:
                d.pieslice((328,372,348,392),90,270,fill=CYAN)
                d.pieslice((328,372,348,392),270,450,fill=CORAL)
            text(d,(370,255 if sign>0 else 493),label,24,col,True)
        text(d,(78,216),'Keep x. Discard y.',25)
        text(d,(353,393),'(1, 0)',24,AMBER,True)
        card(d,(738,199,1232,566),'The label is lost',AMBER)
        paragraph(d,(764,277),'(1, +1) → (1, 0)\n(1, −1) → (1, 0)',440,29,line_height=48)
        paragraph(d,(764,399),'Same projected point; different original labels.',435,28,line_height=39)
        text(d,(764,514),'Hand-chosen projection, not PCA.',22,MUTED)
    else:
        for y,col,number,label,detail in [(211,CYAN,'1','Translation','Distances stay fixed; cosine can move.'),
                    (324,CORAL,'2','Fixed readout','Unchanged geometry can change a decision.'),
                    (437,AMBER,'3','Projection','Discarding a direction can erase a label.')]:
            box(d,(48,y,1232,y+96),outline=GRID,radius=13)
            dot(d,(93,y+49),col,22);text(d,(93,y+49),number,24,BG,True,anchor='mm')
            text(d,(140,y+15),label,27,col,True);text(d,(140,y+54),detail,25)


def content(d,idx,u):
    if idx==0:
        labels=[('WHERE','Bay 3'),('WHEN','18:45'),('REQUIRED','printed ID')]
        for i,(label,value) in enumerate(labels):
            x=48+i*402;card(d,(x,211,x+380,443),label,CYAN)
            text(d,(x+24,310),value,42,INK,True)
        text(d,(78,490),'A requested format changes wording, not the supplied fact ledger.',27)
        text(d,(78,538),'This is a fixed formatter, not a trained language model.',24,MUTED)
    elif idx==1:
        for i,register in enumerate(c.REGISTERS):
            y=201+i*118;box(d,(48,y,1232,y+103),outline=CYAN if u >= i*1.45 else GRID,radius=12)
            if u < i*1.45:continue
            text(d,(72,y+12),register.upper(),19,CYAN,True)
            text(d,(72,y+49),c.format_facts(c.FACTS,register),26)
        text(d,(75,562),'Same literals: Bay 3 · 18:45 · printed ID',21,AMBER)
    elif idx==2:
        card(d,(48,208,1232,382),'Shorter, but incomplete',CORAL)
        text(d,(76,290),'“Check in this evening. Bring your ID.”',36)
        for i,(label,val) in enumerate([('PLACE','Bay 3'),('EXACT TIME','18:45'),('COPY REQUIRED','printed ID')]):
            x=48+i*402;box(d,(x,413,x+380,556),outline=CORAL)
            text(d,(x+23,433),label,19,MUTED,True);text(d,(x+23,476),f'Missing: {val}',29,CORAL,True)
    else:
        card(d,(48,208,623,448),'Wrong supplied place',CORAL)
        text(d,(76,287),'Bay 9',48,CORAL,True)
        text(d,(76,371),'Every template repeats Bay 9.',25)
        card(d,(650,208,1232,448),'Agreement ≠ truth',AMBER)
        paragraph(d,(678,280),'Literal retention does not verify the ledger or test understanding.',510,29,line_height=42)
        text(d,(78,490),'Check separately: preservation · factual correctness · comprehension',26,AMBER)
        text(d,(78,537),'No auxiliary-head benefit is demonstrated by this example.',24,MUTED)


def memory_bar(d,x,y,include_overhead=True,progress=1):
    m=c.mesh_numbers();scale=19
    parts=[('Weights',m['weights_gib'],CYAN)]
    if include_overhead:parts +=[('20% overhead',m['weights_gib']*.2,CORAL),('KV',2.5,AMBER),('Work',2,GREEN)]
    current=x
    for label,value,color in parts:
        extent=min(value*scale,max(0,progress*m['envelope_gib']*scale-(current-x)))
        if extent>0:d.rectangle((current,y,current+extent,y+52),fill=color)
        current+=value*scale
    return current


def mesh(d,idx,u):
    m=c.mesh_numbers()
    if idx==0:
        text(d,(72,210),'70 billion parameters × 4 bits ÷ 8',32,INK,True)
        text(d,(72,260),'35.00 GB = 32.596 GiB ideal weights',29,CYAN,True)
        memory_bar(d,74,320,progress=ease((u-.4)/3.0))
        text(d,(930,329),'43.616 GiB',30,AMBER,True)
        legends=[(74,CYAN,'32.596 weights'),(384,CORAL,'6.519 overhead'),(700,AMBER,'2.500 KV'),(967,GREEN,'2.000 work')]
        for x,col,label in legends:
            d.rectangle((x,399,x+16,415),fill=col);text(d,(x+26,394),label,22)
        text(d,(74,453),'Envelope = 1.20 × weights + KV + 2 GiB',27)
        text(d,(74,501),'KV: 80 layers · 8 KV heads · 128 dims · 8,192 tokens · batch 1 · FP16',23,MUTED)
        text(d,(74,542),'Overhead, workspace and allocatable budgets are illustrative assumptions.',22,MUTED)
    elif idx==1:
        text(d,(74,209),'Assume 12 GiB simultaneously allocatable per pool.',29)
        for j,(label,count,needed,col) in enumerate([('Weights only',3,m['weights_gib'],CYAN),('Runtime envelope',4,m['envelope_gib'],AMBER)]):
            y=281+j*130;text(d,(74,y+1),label,26,col,True)
            for i in range(count):
                x=407+i*145;box(d,(x,y-2,x+129,y+61),fill=PANEL,outline=col,radius=8)
                text(d,(x+64,y+30),'12 GiB',24,INK,True,anchor='mm')
            text(d,(1018,y+12),f'floor: {count}',29,col,True)
            text(d,(408,y+75),f'ceil({needed:.3f} / 12) = {count}',23,MUTED)
        text(d,(74,546),'Aggregate capacity ≠ feasible placement. Pool count ≠ stage count.',25,CORAL,True)
    elif idx==2:
        text(d,(74,207),'One serial boundary',28,CYAN,True)
        text(d,(74,252),'20 ms + 16,384 B / (2,500,000 B/s) = 26.5536 ms',29)
        for i in range(4):
            x=75+i*306;box(d,(x,336,x+208,427),outline=CYAN,radius=10)
            text(d,(x+104,382),f'Stage {i+1}',27,INK,True,anchor='mm')
            if i<3:arrow(d,(x+221,381),(x+289,381),CORAL,4)
        q=ease((u-1)/3.3);positions=[(179,382),(485,382),(791,382),(1097,382)]
        pos=min(2,int(q*3));fraction=q*3-pos if q<1 else 1
        point=(positions[pos][0]+306*fraction,454)
        dot(d,point,AMBER,10)
        text(d,(74,484),'4 serial stages → 3 crossings → 79.6608 ms communication alone',26,AMBER)
        text(d,(74,536),'Motion shows dependency order, not measured packet timing.',23,MUTED)
    else:
        text(d,(74,207),'Tcomm = (H − 1) × (latency + payload / byte rate)',29)
        text(d,(74,245),'Declared one-token serial decoding graph; not all orchestration.',21,MUTED)
        for x,Hh,time,ceiling,col in [(48,4,m['serial_4_s'],m['ceiling_4'],CYAN),(650,16,m['serial_16_s'],m['ceiling_16'],CORAL)]:
            card(d,(x,268,x+582,487),f'{Hh} sequential stages',col)
            text(d,(x+25,334),f'{Hh-1} boundaries → {time:.7f} s',28)
            text(d,(x+25,388),f'≤ {ceiling:.3f} token/s',35,col,True)
            text(d,(x+25,446),'Communication-only ceiling; not achieved rate.',21,MUTED)
        text(d,(73,518),'Excludes compute, queues, protocol, return traffic, relay, contention and failures.',23,AMBER)
        text(d,(73,554),'Overlapping independent requests is a different throughput question.',22,MUTED)


def timeout(d,idx,u):
    if idx==0:
        card(d,(48,225,365,489),'Caller',CYAN);card(d,(870,225,1232,489),'Toy service',CORAL)
        text(d,(75,307),'submit op-1',28);text(d,(75,351),'“add A”',28)
        request=ease((u-.3)/1.1)
        if request>0:arrow(d,(398,289),(398+438*request,289),CYAN,4)
        text(d,(609,247),'request',24,CYAN,anchor='mt')
        if u>=2.7:arrow(d,(837,416),(837-436*ease((u-2.7)/1.1),416),CORAL,4)
        if u>=3.3:text(d,(615,449),'reply lost',27,CORAL,True,anchor='mt')
        if u>=3.3:
            d.line((588,396,628,436),fill=CORAL,width=5);d.line((628,396,588,436),fill=CORAL,width=5)
        text(d,(896,307),'commit: add A' if u>=1.6 else 'await request',26)
        text(d,(896,364),'1 hidden effect' if u>=1.6 else '0 hidden effects',30,AMBER,True)
        text(d,(76,529),'Caller observes: reply unavailable' if u>=3.5 else 'Caller: waiting for a reply…',28,AMBER,True)
        text(d,(761,532),'Effect count: evaluator only',20,MUTED)
    elif idx==1:
        for x,label,count,col in [(48,'No initial commit',0,CYAN),(650,'Commit + lost reply',1,CORAL)]:
            card(d,(x,216,x+582,497),label,col)
            text(d,(x+26,292),f'Hidden effects: {count}',32,col,True)
            text(d,(x+26,353),'Caller sees:',24,MUTED)
            box(d,(x+24,397,x+551,464),fill=BG,radius=8)
            text(d,(x+46,414),'reply unavailable',29,AMBER,True)
        text(d,(75,538),'Identical observations do not identify the hidden history.',28)
    elif idx==2:
        for x,label,count,detail,col in [(48,'ID only',2,'No deduplication',CORAL),(450,'Durable key',1,'Same payload, retained key',GREEN),(852,'Expired key',2,'Protection ended at retry',CORAL)]:
            card(d,(x,211,x+380,488),label,col)
            text(d,(x+25,278),'After one retry' if u>=2.2 else 'Before the retry',23,MUTED)
            text(d,(x+25,322),str(count if u>=2.2 else 1),66,col,True)
            text(d,(x+87,351),'hidden effects',24)
            paragraph(d,(x+25,426),detail,330,23,line_height=31)
        text(d,(75,523),'All three start with a committed write, then a lost reply.',26)
        text(d,(75,563),'No initial commit + retry: 1 effect. Counts are simulator ground truth.',21,MUTED)
    else:
        for y,num,label,detail,col in [(209,'1','Inspect exact evidence','Does this operation have a reliable committed result?',CYAN),
                                     (324,'2','Check the contract','Payload scope, key retention, freshness and late commits.',AMBER),
                                     (439,'3','Preserve unknown','No reliable resolution? Do not invent success or failure.',CORAL)]:
            box(d,(48,y,1232,y+97),outline=GRID,radius=12)
            dot(d,(94,y+49),col,23);text(d,(94,y+49),num,24,BG,True,anchor='mm')
            text(d,(142,y+12),label,27,col,True);text(d,(142,y+53),detail,25)


DRAWERS={'rehearsal-selection':replay,'coordinate-readout':geometry,'content-explanation':content,
         'capacity-latency':mesh,'retry-idempotency':timeout}


def frame(scene,t):
    im,d,idx,u=common(scene,t);DRAWERS[scene](d,idx,u);return im


def vtt_time(seconds):return f'00:00:{seconds:02d}.000'


def write_text_files(scene,out):
    story=c.STORIES[scene]
    transcript=[story['title'],story['badge'],'','No audio. All meaning is available in this transcript and on-screen text.',
                'The MP4 has no autoplay; the HTML viewer defaults to a static poster and respects reduced motion.','']
    captions=['WEBVTT','']
    for i,s in enumerate(story['scenes']):
        transcript += [f"{s['start']:02d}–{s['end']:02d} seconds: {s['label']}",s['caption'],s['description'],'']
        captions += [str(i+1),f"{vtt_time(s['start'])} --> {vtt_time(s['end'])}",s['caption'],'']
    (out/'transcript.txt').write_text('\n'.join(transcript),encoding='utf8')
    (out/'captions.vtt').write_text('\n'.join(captions),encoding='utf8')
    title_=html.escape(story['title']);alt=html.escape(story['poster_alt'],quote=True)
    sections='\n'.join(f'<section><h2>{s["start"]:02d}–{s["end"]:02d}s: {html.escape(s["label"])}</h2><p>{html.escape(s["description"])}</p></section>' for s in story['scenes'])
    page=f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_}</title><style>
:root {{color-scheme:dark}} body {{font:18px/1.6 system-ui,sans-serif;max-width:1040px;margin:auto;padding:24px;background:#0c1422;color:#f0f4fc}}
a {{color:#54d8df}} img,video {{width:100%;height:auto;border-radius:12px}} button {{padding:12px 18px;font:inherit;border:2px solid #54d8df;background:#142238;color:inherit;border-radius:8px;cursor:pointer}}
[hidden] {{display:none!important}} h1{{font-size:clamp(26px,4vw,40px);line-height:1.2}} h2{{font-size:22px}} .note{{color:#ffd285}} :focus-visible{{outline:3px solid #ffd285;outline-offset:4px}}
@media(prefers-reduced-motion:reduce) {{*{{scroll-behavior:auto!important;animation:none!important;transition:none!important}}}}
</style><main><h1>{title_}</h1><p class="note">{html.escape(story['badge'])}</p>
<p>24 seconds, no audio. Nothing plays automatically. The static poster and complete text are the reduced-motion path.</p>
<img id="poster" src="poster.png" alt="{alt}"><video id="clip" controls playsinline preload="none" poster="poster.png" hidden aria-label="{title_}"><source src="concept.mp4" type="video/mp4"><track kind="captions" src="captions.vtt" srclang="en" label="English" default>Your browser cannot play this MP4. Read the transcript below.</video>
<p><button id="play" type="button">Choose to play animation</button> <button id="still" type="button" hidden>Return to static poster</button></p>
<p id="motion-note"></p><p><a href="concept.mp4">Download MP4</a> · <a href="concept.gif">Optional animated GIF</a> · <a href="poster.png">Static PNG</a> · <a href="transcript.txt">Plain-text transcript</a> · <a href="captions.vtt">Captions</a></p>
<h2>Complete transcript and visual description</h2>{sections}</main>
<script>
const poster=document.querySelector('#poster'),clip=document.querySelector('#clip'),play=document.querySelector('#play'),still=document.querySelector('#still');
const preference=matchMedia('(prefers-reduced-motion: reduce)');
function note(){{document.querySelector('#motion-note').textContent=preference.matches?'Reduced motion is requested. Keep the static poster, or explicitly choose to play.':'Static by default. Playback is always your choice.'}}
note();preference.addEventListener?.('change',()=>{{clip.pause();clip.hidden=true;poster.hidden=false;play.hidden=false;still.hidden=true;note()}});
play.addEventListener('click',()=>{{poster.hidden=true;clip.hidden=false;play.hidden=true;still.hidden=false;clip.focus();clip.play().catch(()=>{{document.querySelector('#motion-note').textContent='Use the video controls to play, or return to the static poster.'}})}});
still.addEventListener('click',()=>{{clip.pause();clip.currentTime=0;clip.hidden=true;poster.hidden=false;play.hidden=false;still.hidden=true;play.focus()}});
</script></html>'''
    (out/'index.html').write_text(page,encoding='utf8')
    snippet=f'''### Watch the concept in 24 seconds\n\n[![{story['poster_alt']}](docs/animations/poster.png)](docs/animations/concept.mp4)\n\n[MP4](docs/animations/concept.mp4) · [Optional GIF preview](docs/animations/concept.gif) · [Text transcript](docs/animations/transcript.txt) · [Captions](docs/animations/captions.vtt)\n\n{story['badge'].capitalize()}. No audio. The poster and transcript provide a no-motion alternative. The downloaded [HTML viewer](docs/animations/index.html) is static by default and never autoplays.\n'''
    (out/'README-SNIPPET.md').write_text(snippet,encoding='utf8')


def render(scene,out,preview_only=False):
    out.mkdir(parents=True,exist_ok=True)
    qa=out/'qa';qa.mkdir(exist_ok=True)
    write_text_files(scene,out)
    frame(scene,23.9).save(out/'poster.png',optimize=True)
    for stamp in [0,3,6,9,12,15,18,21,23.9]:frame(scene,stamp).save(qa/f'frame-{stamp:04.1f}.png',optimize=True)
    if preview_only:return
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(c.FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','23',
             '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-metadata','comment=Calculated teaching illustration; not empirical performance evidence',str(out/'concept.mp4')]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for n in range(c.DURATION*c.FPS):process.stdin.write(frame(scene,n/c.FPS).tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('ffmpeg MP4 render failed')
    except Exception:
        process.kill();process.wait();raise
    # Optional browser-friendly preview, 6 fps and 768px wide; not the default display path.
    gif=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out/'concept.mp4'),'-filter_complex',
         'fps=6,scale=768:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96:stats_mode=diff[p];[s1][p]paletteuse=dither=none',
         '-threads','2','-filter_complex_threads','1','-loop','0',str(out/'concept.gif')]
    subprocess.run(gif,check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(out/'concept.mp4')]))
    (qa/'ffprobe.json').write_text(json.dumps(probe,indent=2)+'\n')
    # Decode actual MP4 frames instead of relying only on source frames.
    for stamp in [0,12,23]:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(stamp),'-i',str(out/'concept.mp4'),'-frames:v','1','-threads','1',str(qa/f'decoded-{stamp:02}.png')],check=True)
    files={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in out.iterdir() if f.is_file()}
    environment={'python':sys.version,'pillow':pillow_version,'platform':platform.platform(),
                 'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
                 'font_files':{Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [REGULAR,BOLD,MONO]}}
    (qa/'render-manifest.json').write_text(json.dumps({'scene':scene,'duration_s':c.DURATION,'fps':c.FPS,'size':[W,H],
        'mp4_command':command,'gif_command':gif,'environment':environment,'files':files,'layout_failures':LAYOUT_FAILURES},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=list(c.STORIES),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args()
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:raise SystemExit('Existing ffmpeg and ffprobe are required; no packages will be installed.')
    render(args.scene,args.out,args.preview_only)
    if LAYOUT_FAILURES:raise SystemExit('Text height overflow: '+json.dumps(LAYOUT_FAILURES))
    print(json.dumps({'scene':args.scene,'output':str(args.out),'rendered':not args.preview_only,'layout_failures':0}))


if __name__=='__main__':main()
