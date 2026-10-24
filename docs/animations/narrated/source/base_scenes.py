"""Narrated-path base scenes adapted from the preserved silent Manim source.

Changes clarify moving-score labels and leave final ceiling reveal to narration.
Original silent source remains unchanged in docs/animations/source/scenes.py.

Designed for manim==0.22.0, Cairo renderer, DejaVu Sans Text (no TeX dependency).
Numbers are exact educational toy calculations, never measured model performance.
"""
from manim import (
    Scene, VGroup, Text, Axes, NumberLine, Dot, Circle, Line, DashedLine,
    Arrow, Rectangle, Cross, ValueTracker, VMobject, TracedPath,
    Create, FadeIn, FadeOut, Transform, TransformFromCopy, ReplacementTransform,
    GrowArrow, GrowFromEdge, MoveAlongPath, Indicate, AnimationGroup,
    always_redraw, config, UP, DOWN, LEFT, RIGHT, ORIGIN, linear, smooth,
)
import manim
import numpy as np
import models as m

config.background_color = '#0b0f15'
config.frame_rate = 15
config.pixel_width = 1280
config.pixel_height = 720
WHITE='#f4f5f7'; GRAY='#8994a2'; CYAN='#5ad8df'; CORAL='#ff9395'; GOLD='#ffd27c'; GREEN='#a0e3b4'
FONT='DejaVu Sans'
DURATION=24
MANIM_VERSION_REQUIRED='0.22.0'


def label(value,size=28,color=WHITE):
    return Text(str(value),font=FONT,font_size=size,color=color,disable_ligatures=True)


def xy_axes(x_range,y_range,x_length=6,y_length=4,center=ORIGIN):
    return Axes(x_range=x_range,y_range=y_range,x_length=x_length,y_length=y_length,
        axis_config={'include_tip':False,'include_numbers':False,'stroke_width':1.4,'color':GRAY}).move_to(center)


def piecewise_path(points,color):
    return VMobject(color=color,stroke_width=4).set_points_as_corners(points)


class MinimalLesson(Scene):
    title=''; evidence='Illustrative calculation · not measured evidence'
    def setup(self):
        if manim.__version__ != MANIM_VERSION_REQUIRED:
            raise RuntimeError(f'Pinned Manim {MANIM_VERSION_REQUIRED} required; got {manim.__version__}')
        self.header=label(self.title,38).to_edge(UP,buff=.35)
        self.evidence_label=label(self.evidence,16,GOLD).to_edge(DOWN,buff=.17)
        self.add(self.header,self.evidence_label)
        self.body=VGroup();self.caption=None
    def at(self,seconds):
        remaining=seconds-self.time
        if remaining < -1/15-1e-5:
            raise RuntimeError(f'Scene overran timed boundary {seconds}: {self.time}')
        if remaining>1e-5:self.wait(remaining)
    def caption_to(self,value):
        if self.caption is not None:self.remove(self.caption)
        self.caption=label(value,24).move_to(DOWN*2.95)
        if self.caption.width>12.8:self.caption.scale_to_fit_width(12.8)
        self.add(self.caption)
    def clear_body(self):
        active=[obj for obj in self.mobjects if obj not in (self.header,self.evidence_label,self.caption)]
        for obj in active:obj.clear_updaters(recursive=True)
        if active:self.play(*[FadeOut(obj) for obj in active],run_time=.4)
        self.body=VGroup()
    def keep(self,*objects):self.body.add(*objects);return objects
    def finish(self):self.at(DURATION)


class ReplaySelection(MinimalLesson):
    title='Learning without losing yesterday'
    evidence='Two-weight linear toy · not a benchmark or biological simulation'
    def construct(self):
        # A is w1=1; B is w1+w2=0. The exact shared solution is (1,-1).
        self.caption_to('Compatible tasks can still interfere through shared weights.')
        ax=xy_axes([0,1.2,.2],[-1.2,.2,.2],6.0,4.0,LEFT*1.8+DOWN*.1)
        a_line=DashedLine(ax.c2p(1,-1.2),ax.c2p(1,.2),color=CYAN)
        b_line=Line(ax.c2p(0,0),ax.c2p(1.2,-1.2),color=CORAL)
        origin=ax.c2p(0,0);learned=ax.c2p(1,0);solution=ax.c2p(1,-1)
        w=Dot(origin,color=WHITE,radius=.075)
        target=Circle(radius=.12,color=GOLD,stroke_width=2).move_to(solution)
        labels=VGroup(label('A: w1 = 1',26,CYAN),label('B: w1 + w2 = 0',26,CORAL),
                      label('Both: (1, −1)',26,GOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.55).move_to(RIGHT*3.1+UP*.55)
        axis_labels=VGroup(label('w1',20,GRAY).move_to(ax.c2p(1.15,-.13)),label('w2',20,GRAY).next_to(ax.y_axis,UP))
        self.keep(ax,a_line,b_line,w,target,labels,axis_labels)
        self.play(Create(ax),FadeIn(axis_labels),run_time=.8)
        self.play(Create(a_line),Create(b_line),FadeIn(labels),FadeIn(target),run_time=1.2)
        self.play(FadeIn(w),run_time=.4)
        self.play(w.animate.move_to(learned),run_time=2.4,rate_func=smooth)
        learned_text=label('after A: (1, 0)',23).next_to(w,UP,buff=.25);self.keep(learned_text)
        self.play(FadeIn(learned_text),run_time=.4)
        self.at(8)
        # Actual discrete SGD states from the accepted demo, not a hand-drawn curve.
        self.caption_to('B only forgets A; rehearsing stored A can retain both in this toy.')
        self.play(FadeOut(learned_text),FadeOut(labels),FadeOut(w),run_time=.4)
        learned_text.clear_updaters();self.body.remove(learned_text,labels,w)
        start=m.replay_results()['after_a']['weights'];states={}
        for name,batch in [('B only',(m.TASK_B,m.TASK_B)),('B + A',(m.TASK_B,m.TASK_A))]:
            state=start;path=[ax.c2p(*state)]
            for _ in range(120):state=m.step(state,batch);path.append(ax.c2p(*state))
            states[name]=path
        red=Dot(learned,color=CORAL,radius=.075);blue=Dot(learned,color=CYAN,radius=.075)
        red_path=piecewise_path(states['B only'],CORAL);blue_path=piecewise_path(states['B + A'],CYAN)
        descriptions=VGroup(label('B only',28,CORAL),label('A prediction → 0.500',24,CORAL),
                            label('B + stored A',28,CYAN),label('A prediction → 0.999',24,CYAN)).arrange(DOWN,aligned_edge=LEFT,buff=.38).move_to(RIGHT*3.15+UP*.4)
        budgets=label('120 updates · 240 presentations each',21,GRAY).move_to(DOWN*2.35)
        self.keep(red,blue,red_path,blue_path,descriptions,budgets)
        self.play(FadeIn(red),FadeIn(blue),FadeIn(descriptions),FadeIn(budgets),run_time=.6)
        self.play(Create(red_path),Create(blue_path),MoveAlongPath(red,red_path),MoveAlongPath(blue,blue_path),run_time=3.4,rate_func=linear)
        memory_note=label('Stores one old example\nUses fewer B presentations',18,GOLD).move_to(RIGHT*3.1+DOWN*1.6)
        self.keep(memory_note);self.play(FadeIn(memory_note),run_time=.4)
        self.at(16)
        self.clear_body();self.caption_to('Hard now is not the same as threatened next. The virtual update is discarded.')
        ax2=xy_axes([0,.6,.1],[-.25,.16,.1],8.4,3.9,DOWN*.12)
        axis_words=VGroup(label('current loss',22,GRAY).move_to(RIGHT*5.0+UP*.25),
                          label('Δloss',22,GRAY).move_to(LEFT*5.3+UP*1.8))
        rows=m.selection_scores();dots=[];names=[]
        for row,col in zip(rows,[GRAY,CORAL,CYAN]):
            dot=Dot(ax2.c2p(row['now'],0),color=col,radius=.09);dots.append(dot)
            names.append(label(row['name'],23,col))
        score_step=label('virtual weights: (1, 0) → (0.75, −0.25)',22).move_to(UP*2.1)
        self.keep(ax2,axis_words,score_step,*dots,*names)
        self.play(Create(ax2),FadeIn(axis_words),FadeIn(score_step),*[FadeIn(d) for d in dots],run_time=.8)
        self.play(*[dot.animate.move_to(ax2.c2p(row['now'],row['delta'])) for dot,row in zip(dots,rows)],run_time=2.4)
        names[0].next_to(dots[0],RIGHT,buff=.2);names[1].next_to(dots[1],UP,buff=.2);names[2].next_to(dots[2],RIGHT,buff=.2)
        self.play(*[FadeIn(n) for n in names],run_time=.6)
        self.play(Indicate(dots[1],color=CORAL),Indicate(dots[2],color=CYAN),run_time=.8)
        self.final_focus=[dots[1],dots[2]]
        self.finish()


class CoordinateReadout(MinimalLesson):
    title='Same distance. Different decisions.'
    evidence='Exact geometric examples · no pretrained-model experiment'
    def construct(self):
        self.caption_to('Translate both vectors: distance stays fixed, cosine changes.')
        ax=xy_axes([0,4,1],[0,4,1],4.5,4.5,LEFT*2+DOWN*.2)
        t=ValueTracker(0)
        a=lambda:ax.c2p(1+t.get_value(),t.get_value())
        b=lambda:ax.c2p(t.get_value(),1+t.get_value())
        va=always_redraw(lambda:Arrow(ax.c2p(0,0),a(),buff=0,color=CYAN,stroke_width=4))
        vb=always_redraw(lambda:Arrow(ax.c2p(0,0),b(),buff=0,color=CORAL,stroke_width=4))
        separation=always_redraw(lambda:Line(a(),b(),color=GOLD,stroke_width=3))
        ad=always_redraw(lambda:Dot(a(),color=CYAN));bd=always_redraw(lambda:Dot(b(),color=CORAL))
        al=always_redraw(lambda:label('A',22,CYAN).next_to(ad,RIGHT,buff=.15))
        bl=always_redraw(lambda:label('B',22,CORAL).next_to(bd,UP,buff=.15))
        distance=label('distance = √2',30,GOLD).move_to(RIGHT*3+UP*1.1)
        cosine=always_redraw(lambda:label(f"cosine = {m.cosine((1+t.get_value(),t.get_value()),(t.get_value(),1+t.get_value())):.3f}",29).move_to(RIGHT*3+DOWN*.05))
        shift=label('+ (2, 2)',28,GRAY).move_to(RIGHT*3+DOWN*1.1)
        geom_axes=VGroup(label('0',17,GRAY).next_to(ax.c2p(0,0),DOWN+LEFT,buff=.1),label('x1',19,GRAY).next_to(ax.x_axis,RIGHT,buff=.1),label('x2',19,GRAY).next_to(ax.y_axis,UP,buff=.1))
        self.add(geom_axes)
        self.keep(ax,va,vb,separation,ad,bd,al,bl,distance,cosine,shift,geom_axes)
        self.play(Create(ax),GrowArrow(va),GrowArrow(vb),FadeIn(ad),FadeIn(bd),FadeIn(al),FadeIn(bl),FadeIn(distance),FadeIn(cosine),run_time=1)
        self.play(FadeIn(shift),Create(separation),run_time=.6)
        self.play(t.animate.set_value(2),run_time=3.6,rate_func=smooth)
        self.at(8)
        self.clear_body();self.caption_to('A fixed readout can flip B. Compensating its bias is a second edit.')
        line=NumberLine(x_range=[-1,3,1],length=10,include_numbers=False,include_tip=False,color=GRAY)
        zero=DashedLine(line.n2p(0)+UP*.7,line.n2p(0)+DOWN*.7,color=GRAY)
        zero_label=label('0',22,GRAY).next_to(zero,DOWN)
        dot=Dot(line.n2p(-.5),color=CORAL,radius=.11)
        bname=label('B',25,CORAL).next_to(dot,DOWN,buff=.25)
        equation=label('score = x1 − 0.5',32).move_to(UP*1.85)
        before=label('−0.5',26,CORAL).next_to(dot,UP,buff=.3)
        target=label('+1.5',26,CORAL).next_to(line.n2p(1.5),UP,buff=.3)
        self.keep(line,zero,zero_label,dot,bname,equation,before,target)
        self.play(Create(line),Create(zero),FadeIn(zero_label),FadeIn(dot),FadeIn(bname),FadeIn(equation),FadeIn(before),run_time=.8)
        self.play(dot.animate.move_to(line.n2p(1.5)),bname.animate.next_to(line.n2p(1.5),DOWN,buff=.25),Transform(before,target),run_time=2)
        compensated=label('score = x1 − 2.5',32,GREEN).move_to(equation)
        prior=label('previous +1.5 → new −0.5',21,GOLD).move_to(UP*.9)
        self.keep(prior);self.play(FadeOut(before),FadeIn(prior),Transform(equation,compensated),run_time=.6)
        restored=label('−0.5',26,GREEN).next_to(line.n2p(-.5),UP,buff=.3)
        self.play(dot.animate.move_to(line.n2p(-.5)),bname.animate.next_to(line.n2p(-.5),DOWN,buff=.25),run_time=1.8)
        self.play(FadeOut(prior),FadeIn(restored),run_time=.4)
        self.at(16)
        self.clear_body();self.caption_to('Projection can merge points with different labels: the answer is gone.')
        ax2=xy_axes([0,2,1],[-1.4,1.4,1],7,4.2,DOWN*.1)
        p=Dot(ax2.c2p(1,1),color=CYAN,radius=.1);q=Circle(radius=.13,color=CORAL,stroke_width=4).move_to(ax2.c2p(1,-1))
        positive=label('+y label',25,CYAN).next_to(p,RIGHT,buff=.25)
        negative=label('−y label',25,CORAL).next_to(q,RIGHT,buff=.25)
        rule=label('P(x, y) = (x, 0)',31).move_to(UP*2.15)
        guide=DashedLine(ax2.c2p(1,-1.1),ax2.c2p(1,1.1),color=GRAY)
        result=label('same point (1, 0)',25,GOLD).next_to(ax2.c2p(1,0),RIGHT,buff=.35).shift(UP*.35)
        self.keep(ax2,p,q,positive,negative,rule,guide,result)
        self.play(Create(ax2),Create(guide),FadeIn(p),FadeIn(q),FadeIn(positive),FadeIn(negative),FadeIn(rule),run_time=.8)
        self.play(p.animate.move_to(ax2.c2p(1,0)),q.animate.move_to(ax2.c2p(1,0)),
                  positive.animate.shift(DOWN*1.1),negative.animate.shift(UP*1.1),run_time=2.8)
        self.play(FadeOut(positive),FadeOut(negative),FadeIn(result),run_time=.6)
        self.final_focus=[p,q]
        self.finish()


class ContentExplanation(MinimalLesson):
    title='Change the wording. Keep the facts.'
    evidence='Fixed formatter illustration · no model learning or reader-comprehension claim'
    def construct(self):
        self.caption_to('Requested presentation changes wording, not the supplied facts.')
        facts=VGroup(label('Bay 3',33,CYAN),label('18:45',33,GOLD),label('printed ID',33,CORAL)).arrange(RIGHT,buff=1.4).move_to(UP*1.85)
        f_label=label('f =',28).next_to(facts,LEFT,buff=.5)
        format_label=label('F(f, requested format)',28,GRAY).move_to(UP*.5)
        self.keep(facts,f_label,format_label)
        self.play(FadeIn(f_label),FadeIn(facts),run_time=.8)
        self.play(FadeIn(format_label),run_time=.6)
        template_rows=VGroup(*[label(m.format_facts(m.FACTS,r),24) for r in m.REGISTERS]).arrange(DOWN,aligned_edge=LEFT,buff=.55).move_to(DOWN*.9)
        for row in template_rows:
            if row.width>12:row.scale_to_fit_width(12)
        self.keep(template_rows)
        for row in template_rows:self.play(FadeIn(row),run_time=1.0)
        self.at(8)
        self.clear_body();self.caption_to('A shorter instruction can lose the exact place, time and requirement.')
        originals=VGroup(label('Bay 3',35,CYAN),label('18:45',35,GOLD),label('printed ID',35,CORAL)).arrange(RIGHT,buff=1.5).move_to(UP*1.35)
        shorter=label('“Check in this evening. Bring your ID.”',31).move_to(DOWN*.25)
        missing=VGroup(label('?',38,CYAN),label('?',38,GOLD),label('ID ≠ printed ID',30,CORAL)).arrange(RIGHT,buff=2.05).move_to(DOWN*1.7)
        self.keep(originals,shorter,missing)
        self.play(FadeIn(originals),run_time=.6)
        self.play(FadeIn(shorter),run_time=.8)
        for b in missing:self.play(FadeIn(b),run_time=.8)
        note=label('Literal checks are not semantic or truth checks.',23,GRAY).move_to(UP*2.25)
        self.keep(note);self.play(FadeIn(note),run_time=.6)
        self.at(16)
        self.clear_body();self.caption_to('Agreement is not truth. Wrong input can remain wrong in every format.')
        correct=label('Bay 3',38,CYAN).move_to(LEFT*3.4+UP*.6)
        wrong=label('Bay 9',38,CORAL).move_to(correct)
        outs=VGroup(*[label('Bay 9',32,CORAL) for _ in range(3)]).arrange(DOWN,buff=.6).move_to(RIGHT*3.6)
        arrows=VGroup(*[Arrow(correct.get_right()+RIGHT*.3,o.get_left()+LEFT*.3,buff=0,color=GRAY,stroke_width=2) for o in outs])
        self.keep(correct,outs,arrows)
        self.play(FadeIn(correct),run_time=.6)
        self.play(FadeOut(correct),run_time=.4)
        self.play(FadeIn(wrong),run_time=.6)
        self.play(*[GrowArrow(a) for a in arrows],run_time=.8)
        self.play(*[FadeIn(o) for o in outs],run_time=1.2)
        equality=label('consistent ≠ verified',30,GOLD).move_to(DOWN*2.1)
        self.keep(equality);self.play(FadeIn(equality),run_time=.6)
        self.final_focus=[arrows,outs]
        self.finish()


class CapacityLatency(MinimalLesson):
    title='More memory is not automatically more speed'
    evidence='Calculated assumptions · not device, network or model measurements'
    def construct(self):
        n=m.mesh_numbers();self.caption_to('Ideal weights are only one term in an illustrative runtime envelope.')
        numberline=NumberLine(x_range=[0,48,12],length=11.5,include_numbers=False,include_tip=False,color=GRAY).move_to(DOWN*.4)
        labels=VGroup(*[label(str(x),21,GRAY).next_to(numberline.n2p(x),DOWN,buff=.2) for x in [0,12,24,36,48]])
        unit=label('GiB',22,GRAY).next_to(numberline,RIGHT)
        calc=label('70B × 4 bits ÷ 8 = 35 GB = 32.596 GiB',27).move_to(UP*1.95)
        self.keep(numberline,labels,unit,calc)
        self.play(Create(numberline),FadeIn(labels),FadeIn(unit),FadeIn(calc),run_time=.8)
        parts=[('weights',n['weights_gib'],CYAN),('20% overhead',.2*n['weights_gib'],CORAL),('KV',2.5,GOLD),('work',2,GREEN)]
        current=0;segments=VGroup()
        for name,value,col in parts:
            width=value*11.5/48
            rect=Rectangle(width=width,height=.55,color=col,fill_color=col,fill_opacity=.8,stroke_width=0)
            rect.move_to(numberline.n2p(current+value/2)+UP*.4)
            segments.add(rect);self.keep(rect)
            self.play(GrowFromEdge(rect,LEFT),run_time=.6)
            current+=value
        total=label('1.20 W + 2.5 GiB KV + 2 GiB = 43.616 GiB',28,GOLD).move_to(UP*.85)
        assumptions=label('KV: 80 layers · 8 heads · 128 dims · 8192 tokens · FP16 · batch 1',18,GRAY).move_to(DOWN*1.5)
        legend=VGroup(*[label(name,20,col) for name,_,col in parts]).arrange(RIGHT,buff=.65).move_to(DOWN*1.0)
        self.keep(total,assumptions,legend);self.play(FadeIn(total),FadeIn(assumptions),FadeIn(legend),run_time=.6)
        self.at(8)
        self.clear_body();self.caption_to('Capacity is an aggregate lower bound, not a feasible shard-placement proof.')
        ax=xy_axes([0,48,12],[0,5,1],9.5,4.1,DOWN*.05)
        stairs=piecewise_path([ax.c2p(0,0),ax.c2p(.001,1),ax.c2p(12,1),ax.c2p(12.001,2),ax.c2p(24,2),
                              ax.c2p(24.001,3),ax.c2p(36,3),ax.c2p(36.001,4),ax.c2p(48,4)],CYAN)
        rule=label('pools ≥ ceil(required GiB / 12)',29).move_to(UP*2.15)
        initial=ax.c2p(n['weights_gib'],3);end=ax.c2p(n['envelope_gib'],4)
        dot=Dot(initial,color=GOLD,radius=.09)
        a=label('32.596 → 3 pools',24,GOLD).move_to(LEFT*2.4+UP*1)
        b=label('43.616 → 4 pools',24,GOLD).move_to(RIGHT*2.2+DOWN*1.5)
        assumption=label('Assume 12 GiB simultaneously allocatable per pool.',21,GRAY).move_to(UP*2.65)
        units=VGroup(label('required GiB',20,GRAY).move_to(RIGHT*4.3+DOWN*1.85),label('pools',20,GRAY).move_to(LEFT*5.25+UP*1.9))
        ticks=VGroup(*[label(str(v),17,GRAY).next_to(ax.c2p(v,0),DOWN,buff=.12) for v in [12,24,36,48]],*[label(str(v),17,GRAY).next_to(ax.c2p(0,v),LEFT,buff=.12) for v in [1,3,4]])
        self.add(units,ticks)
        self.keep(ax,stairs,rule,dot,a,b,assumption,units,ticks)
        self.play(Create(ax),Create(stairs),FadeIn(rule),FadeIn(assumption),FadeIn(dot),FadeIn(a),run_time=1.0)
        path=piecewise_path([initial,ax.c2p(36,3),ax.c2p(36.001,4),end],GOLD)
        self.play(MoveAlongPath(dot,path),run_time=2.8,rate_func=linear)
        self.play(FadeIn(b),run_time=.6)
        self.at(16)
        self.clear_body();self.caption_to('A declared one-token serial graph adds dependent crossings; this is not all orchestration.')
        ax2=xy_axes([1,16,3],[0,425,100],9.5,4,DOWN*.1)
        cost=Line(ax2.c2p(1,0),ax2.c2p(16,1000*n['serial_16_s']),color=CORAL,stroke_width=3)
        tracker=ValueTracker(4)
        point=always_redraw(lambda:Dot(ax2.c2p(round(tracker.get_value()),(round(tracker.get_value())-1)*1000*n['crossing_s']),color=GOLD,radius=.09))
        formula=label('Tcomm = (H − 1) × 26.5536 ms',29).move_to(UP*2.15)
        inputs=label('16 KiB · 20 Mbps · 20 ms one-way latency',21,GRAY).move_to(UP*2.65)
        data=always_redraw(lambda:label(f"H={round(tracker.get_value())}   Tcomm={(round(tracker.get_value())-1)*1000*n['crossing_s']:.3f} ms",25,GOLD).move_to(LEFT*.5+UP*1.35))
        units=VGroup(label('stages H',20,GRAY).move_to(RIGHT*4.5+DOWN*1.85),label('communication ms',19,GRAY).rotate(np.pi/2).move_to(LEFT*6.0+UP*.2))
        ticks=VGroup(*[label(str(v),17,GRAY).next_to(ax2.c2p(v,0),DOWN,buff=.1) for v in [1,4,8,12,16]],*[label(str(v),17,GRAY).next_to(ax2.c2p(1,v),LEFT,buff=.1) for v in [100,200,400]])
        self.add(units,ticks)
        self.keep(ax2,cost,point,formula,inputs,data,units,ticks)
        self.play(Create(ax2),Create(cost),FadeIn(point),FadeIn(formula),FadeIn(inputs),FadeIn(data),run_time=1.0)
        self.play(tracker.animate.set_value(16),run_time=3.2,rate_func=linear)
        ceiling=VGroup(label('≤ 2.511 token/s',28,CORAL),label('communication-only ceiling',20,GRAY)).arrange(DOWN,buff=.2).move_to(RIGHT*2.2+DOWN*1.1)
        self.narrated_ceiling=ceiling
        self.finish()


class RetryIdempotency(MinimalLesson):
    title='A lost reply is not a failed write'
    evidence='Atomic sequential toy contracts · not real API guarantees'
    def construct(self):
        self.caption_to('The service can commit even when its reply never reaches the caller.')
        caller=Dot(LEFT*4.5+UP*.7,color=CYAN,radius=.11);service=Dot(RIGHT*4.5+UP*.7,color=CORAL,radius=.11)
        names=VGroup(label('caller',27,CYAN).next_to(caller,UP,buff=.4),label('service',27,CORAL).next_to(service,UP,buff=.4))
        request=Arrow(caller.get_center()+RIGHT*.3,service.get_center()+LEFT*.3,buff=0,color=CYAN)
        op=label('op-1: add A',25).move_to(UP*1.4)
        effect=Circle(radius=.28,color=GOLD,fill_color=GOLD,fill_opacity=.8).move_to(RIGHT*4.5+DOWN*.65)
        effect_label=label('1 hidden effect',23,GOLD).next_to(effect,DOWN,buff=.25)
        reply=Arrow(service.get_center()+DOWN*1.5,LEFT*.2+DOWN*.8,buff=0,color=CORAL)
        lost=Cross(stroke_color=CORAL,stroke_width=4).scale(.22).move_to(LEFT*.3+DOWN*.8)
        timeout=label('reply unavailable',28).move_to(LEFT*3.7+DOWN*1.45)
        self.keep(caller,service,names,request,op,effect,effect_label,reply,lost,timeout)
        self.play(FadeIn(caller),FadeIn(service),FadeIn(names),run_time=.8)
        self.play(GrowArrow(request),FadeIn(op),run_time=1.2)
        self.play(FadeIn(effect),FadeIn(effect_label),run_time=.6)
        self.play(GrowArrow(reply),run_time=1.2)
        self.play(FadeIn(lost),FadeIn(timeout),run_time=.6)
        self.at(8)
        self.clear_body();self.caption_to('The same timeout can hide zero or one initial commit.')
        zero=Circle(radius=.35,color=GRAY,stroke_width=3).move_to(LEFT*3.8+UP*.85)
        one=Circle(radius=.35,color=GOLD,fill_color=GOLD,fill_opacity=.8).move_to(RIGHT*3.8+UP*.85)
        z_label=label('no commit',27).next_to(zero,UP,buff=.35);o_label=label('commit',27).next_to(one,UP,buff=.35)
        t_left=label('timeout',29,CORAL).move_to(LEFT*3.8+DOWN*.85);t_right=label('timeout',29,CORAL).move_to(RIGHT*3.8+DOWN*.85)
        arrows=VGroup(Arrow(zero.get_bottom(),t_left.get_top(),buff=.25,color=GRAY),Arrow(one.get_bottom(),t_right.get_top(),buff=.25,color=GRAY))
        equal=label('same observation',28,GRAY).move_to(DOWN*1.95)
        self.keep(zero,one,z_label,o_label,t_left,t_right,arrows,equal)
        self.play(FadeIn(zero),FadeIn(one),FadeIn(z_label),FadeIn(o_label),run_time=.8)
        self.play(*[GrowArrow(a) for a in arrows],run_time=.8)
        self.play(FadeIn(t_left),FadeIn(t_right),FadeIn(equal),run_time=.8)
        self.play(Indicate(t_left),Indicate(t_right),run_time=1)
        self.at(16)
        self.clear_body();self.caption_to('Same-ID retries depend on the contract. Otherwise preserve unknown.')
        row_names=['ID only','Durable key','Expired key'];counts=m.retry_cases()[:3]
        names=VGroup(*[label(name,27) for name in row_names]).arrange(DOWN,buff=.9).move_to(LEFT*4.2)
        attempt=label('op-1  →  retry op-1',28,GRAY).move_to(UP*2.15)
        first=[];second=[]
        for i,count in enumerate(counts):
            y=names[i].get_y()
            first.append(Circle(radius=.19,color=GOLD,fill_color=GOLD,fill_opacity=.85).move_to(RIGHT*.5+UP*y))
            if count==2:second.append(Circle(radius=.19,color=CORAL,fill_color=CORAL,fill_opacity=.85).move_to(RIGHT*1.3+UP*y))
        count_labels=VGroup(*[label(str(count)+' effect'+('s' if count!=1 else ''),27,GREEN if count==1 else CORAL).move_to(RIGHT*4.15+UP*names[i].get_y()) for i,count in enumerate(counts)])
        self.keep(names,attempt,*first,*second,count_labels)
        self.play(FadeIn(names),FadeIn(attempt),*[FadeIn(x) for x in first],run_time=.8)
        self.play(*[TransformFromCopy(first[i],second[j]) for j,i in enumerate([0,2])],run_time=1.8)
        self.play(FadeIn(count_labels),run_time=.6)
        note=label('IDs label requests; retained payload-bound keys promise deduplication.',20,GOLD).move_to(DOWN*2.15)
        self.keep(note);self.play(FadeIn(note),run_time=.6)
        self.finish()

