"""Additive audio-led rerenders of the preserved educational Manim scenes.

Set NARRATED_PLAN to a measured timeline JSON created by timeline.build_timeline.
The legacy source is retained byte-for-byte. No research result is regenerated.
"""
import json
import math
import os
from pathlib import Path
import base_scenes as legacy
from manim import config
from math_svg import math_mobject

# Manim CLI controls preview dimensions; the legacy module's defaults are not used.
config.frame_rate=30
config.pixel_width=int(os.environ.get('NARRATED_WIDTH','1280'))
config.pixel_height=int(os.environ.get('NARRATED_HEIGHT','720'))

FORMULAS={
    'A: w1 = 1':r'A:\ w_1=1',
    'B: w1 + w2 = 0':r'B:\ w_1+w_2=0',
    'Both: (1, −1)':r'\mathrm{Both}:\ (1,-1)',
    'distance = √2':r'd=\sqrt{2}',
    'score = x1 − 0.5':r's=x_1-0.5',
    'score = x1 − 2.5':r's=x_1-2.5',
    'P(x, y) = (x, 0)':r'P(x,y)=(x,0)',
    'pools ≥ ceil(required GiB / 12)':r'N\geq\lceil M/(12\,\mathrm{GiB})\rceil',
    'Tcomm = (H − 1) × 26.5536 ms':r'T_{\mathrm{comm}}=(H-1)\times26.5536\,\mathrm{ms}',
    'Δloss':r'\Delta L',
    'w1':r'w_1','w2':r'w_2','x1':r'x_1','x2':r'x_2',
}
_prose_label=legacy.label
def label(value,size=28,color=legacy.WHITE):
    if value in FORMULAS:
        obj=math_mobject(FORMULAS[value],size=size/48,color=color)
        return obj.move_to(legacy.ORIGIN)
    return _prose_label(value,size,color)
legacy.label=label

CONCLUSIONS={
    'learning-and-remembering':'Toy mechanism · controlled studies remain separate',
    'language-for-the-reader':'Preservation · truth · comprehension: test each separately',
    'household-compute':'Conditional calculation · other execution graphs need separate analysis',
    'what-activations-do':'Check geometry, decisions, and task information separately',
    'after-the-timeout':'If the evidence cannot settle it, preserve unknown',
}

class AudioLed:
    def setup(self):
        path=Path(os.environ['NARRATED_PLAN'])
        self.narration=json.loads(path.read_text())
        self.sections=self.narration['segments']
        if len(self.sections)!=4 or self.narration['fps']!=30:
            raise ValueError('Expected four measured sections at 30 fps')
        self.phase=0
        super().setup()

    def play(self,*animations,**kwargs):
        # Re-render every frame with changed run_time, never stretch encoded video.
        if 'run_time' in kwargs and self.phase<3 and not getattr(self,'exact_timing',False):
            section=self.sections[self.phase]
            scale=(section['end']-section['start'])/8
            kwargs['run_time']=max(1,round(kwargs['run_time']*scale*30))/30
        return super().play(*animations,**kwargs)

    def at(self,seconds):
        if seconds not in (8,16,24):
            raise ValueError('Unexpected legacy boundary')
        index=int(seconds/8)-1
        target=self.sections[index]['end']
        remaining=target-self.time
        if remaining < -1/30-1e-6:
            raise RuntimeError(f'Animation overran measured narration: {self.time} > {target}')
        if remaining>1e-6:self.wait((round(remaining*30)+1e-6)/30)
        self.phase=index+1

    def finish(self):
        self.at(24)
        # Retain the argument, but guide attention during the spoken conclusion.
        repo=self.narration['repo'];start=self.sections[3]['start'];end=self.sections[3]['end']
        def cue(fraction):
            target=start+fraction*(end-start)
            if target>self.time:self.wait((round((target-self.time)*30)+1e-6)/30)
        if repo=='household-compute':
            self.caption_to('An optimistic communication-only ceiling under the stated assumptions.')
            self.play(legacy.FadeIn(self.narrated_ceiling),run_time=.8)
            cue(.42)
            omitted=label('Compute, queueing, return traffic, and other costs omitted',16,legacy.GOLD).move_to(legacy.DOWN*2.58)
            self.play(legacy.FadeIn(omitted),run_time=.6)
            cue(.72)
            self.caption_to('Different execution graphs need their own analysis.')
        else:
            focus=getattr(self,'final_focus',[])
            if focus:self.play(legacy.Indicate(focus[0],color=legacy.GOLD),run_time=.8)
            self.caption_to({'learning-and-remembering':'These are explanatory toys, not evidence of a winning selector.',
                'language-for-the-reader':'Fixed templates preserve the supplied facts, including mistakes.',
                'what-activations-do':'Exact geometry illustration · not a pretrained-model experiment'}[repo])
            cue(.36)
            if len(focus)>1:self.play(legacy.Indicate(focus[1],color=legacy.GOLD),run_time=.8)
            cue(.62)
            self.caption_to(CONCLUSIONS[repo])
        remaining=end-self.time
        if remaining>0:self.wait((round(remaining*30)+1e-6)/30)

class NarratedReplay(AudioLed,legacy.ReplaySelection):pass
class NarratedReader(AudioLed,legacy.ContentExplanation):pass
class NarratedCapacity(AudioLed,legacy.CapacityLatency):pass
class NarratedGeometry(AudioLed,legacy.CoordinateReadout):pass
class NarratedRecovery(AudioLed,legacy.RetryIdempotency):
    def construct(self):
        # Reuse the original verified first two phases, then hand off before phase three.
        self.recovery_custom=False
        try:
            super().construct()
        except RecoveryHandoff:
            pass

    def clear_body(self):
        if self.phase==2 and not self.recovery_custom:
            self.recovery_custom=True
            self.recovery_comparison()
            # End the inherited construct without running its simultaneous third phase.
            raise RecoveryHandoff()
        return super().clear_body()

    def recovery_comparison(self):
        super().clear_body()
        self.exact_timing=True
        self.caption_to('Same-ID retries depend on the retained key contract.')
        start=self.sections[2]['start']; duration=self.sections[2]['end']-start
        def cue(fraction):
            target=start+fraction*duration
            if target>self.time:self.wait((round((target-self.time)*30)+1e-6)/30)
        names=legacy.VGroup(*[label(x,27) for x in ['ID only','Durable key','Expired key']]).arrange(legacy.DOWN,buff=.9).move_to(legacy.LEFT*4.2)
        self.attempt=label('op-1  →  retry op-1',28,legacy.GRAY).move_to(legacy.UP*2.15)
        self.play(legacy.FadeIn(self.attempt),run_time=.4)
        counts=legacy.m.retry_cases()[:3]
        for index,(fraction,count) in enumerate(zip([.16,.44,.77],counts)):
            cue(fraction)
            name=names[index];y=name.get_y()
            first=legacy.Circle(radius=.19,color=legacy.GOLD,fill_color=legacy.GOLD,fill_opacity=.85).move_to(legacy.RIGHT*.5+legacy.UP*y)
            self.play(legacy.FadeIn(name),legacy.FadeIn(first),run_time=.4)
            if count==2:
                second=legacy.Circle(radius=.19,color=legacy.CORAL,fill_color=legacy.CORAL,fill_opacity=.85).move_to(legacy.RIGHT*1.3+legacy.UP*y)
                self.play(legacy.TransformFromCopy(first,second),run_time=.8)
            else:
                self.play(legacy.Indicate(first,color=legacy.GREEN),run_time=.8)
            result=label(str(count)+' effect'+('s' if count!=1 else ''),27,legacy.GREEN if count==1 else legacy.CORAL).move_to(legacy.RIGHT*4.15+legacy.UP*y)
            self.play(legacy.FadeIn(result),run_time=.3)
        note=label('Retained, durable, payload-bound key',22,legacy.GOLD).move_to(legacy.DOWN*2.05)
        self.play(legacy.FadeIn(note),run_time=.4)
        self.at(24)
        self.play(legacy.FadeOut(self.attempt),run_time=.3)
        scope=legacy.VGroup(*[label(x,21,legacy.GOLD) for x in ['Atomic calls','Sequential caller','No late original']]).arrange(legacy.RIGHT,buff=.55).move_to(legacy.UP*2.15)
        for item in scope:self.play(legacy.FadeIn(item),run_time=.4)
        conclusion_start=self.sections[3]['start'];conclusion_end=self.sections[3]['end']
        self.caption_to('Real recovery also needs freshness, expiry, and in-flight status.')
        midpoint=conclusion_start+.58*(conclusion_end-conclusion_start)
        if self.time<midpoint:self.wait((round((midpoint-self.time)*30)+1e-6)/30)
        self.caption_to('If the evidence cannot settle it, preserve unknown.')
        self.wait((round((conclusion_end-self.time)*30)+1e-6)/30)



class RecoveryHandoff(Exception):
    """Internal control flow: the custom final phases have completed."""
    pass
