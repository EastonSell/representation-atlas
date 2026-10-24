"""Original educational animation data, adapted from the repositories' learning demos.

All numbers are deterministic toy calculations or explicitly assumed inputs.
Nothing here runs a model, discovers hardware, or contacts a service.
"""
import math

GIB = 1_073_741_824
TASK_A = ((1.,0.),1.)
TASK_B = ((1.,1.),0.)
RATE = .25


def loss(weights, example):
    x,y=example
    return .5*(sum(w*v for w,v in zip(weights,x))-y)**2


def step(weights, examples):
    gradient=[0.,0.]
    for x,y in examples:
        error=sum(w*v for w,v in zip(weights,x))-y
        for i in range(2): gradient[i] += error*x[i]/len(examples)
    return tuple(w-RATE*g for w,g in zip(weights,gradient))


def replay_results():
    w=(0.,0.)
    for _ in range(40): w=step(w,(TASK_A,TASK_A))
    out={'after_a':{'weights':w,'a_loss':loss(w,TASK_A),'b_loss':loss(w,TASK_B)}}
    for name,batch in [('current_only',(TASK_B,TASK_B)),('replay',(TASK_B,TASK_A))]:
        v=w
        for _ in range(120):v=step(v,batch)
        out[name]={'weights':v,'a_loss':loss(v,TASK_A),'b_loss':loss(v,TASK_B),
                   'updates':120,'presentations':240,'stored_old_examples':int(name=='replay')}
    return out


def selection_scores():
    w=(1.,0.); virtual=step(w,(TASK_B,))
    items=[('Old anchor',TASK_A),('Hard now',((0.,1.),-1.)),('Threatened next',((2.,0.),2.))]
    return [{'name':n,'now':loss(w,e),'after':loss(virtual,e),'delta':loss(virtual,e)-loss(w,e)} for n,e in items]


def cosine(a,b):
    return sum(x*y for x,y in zip(a,b))/(math.hypot(*a)*math.hypot(*b))


def readout(point,bias):return point[0]+bias

def project(point):return (point[0],0)

REGISTERS=('introductory','working','technical')
FACTS={'place':'Bay 3','time':'18:45','requirement':'printed ID'}


def format_facts(facts,register):
    return {'introductory':'Go to {place} at {time}. Bring {requirement}.',
            'working':'Check-in: {place}, {time}. Required: {requirement}.',
            'technical':'Location={place}; time={time}; requirement={requirement}.'}[register].format(**facts)


def mesh_numbers():
    w=70_000_000_000*4//8
    kv=2*80*8*128*8192*2
    envelope=(w*6+4)//5+kv+2*GIB
    crossing=.020+16384/(20_000_000/8)
    return {'weights_bytes':w,'weights_gib':w/GIB,'kv_bytes':kv,'envelope_gib':envelope/GIB,
            'weight_pools':math.ceil(w/(12*GIB)),'envelope_pools':math.ceil(envelope/(12*GIB)),
            'crossing_s':crossing,'serial_4_s':3*crossing,'serial_16_s':15*crossing,
            'ceiling_4':1/(3*crossing),'ceiling_16':1/(15*crossing)}


class LostReply(Exception):pass


class ToyService:
    """Atomic sequential toy only; retained keys bind a payload, effects survive expiry."""
    def __init__(self,deduplicates=True,lifetime=None):
        self.deduplicates=deduplicates;self.lifetime=lifetime;self.effects=[];self.keys={}
    def submit(self,operation_id,payload,*,now,lose=None):
        if lose=='before':raise LostReply('reply unavailable')
        cached=self.keys.get(operation_id) if self.deduplicates else None
        if cached is not None:
            original,result,expires=cached
            if expires is None or now < expires:
                if original!=payload:raise ValueError('Retained key conflicts with a different payload')
                if lose=='after':raise LostReply('reply unavailable')
                return result
        self.effects.append((operation_id,payload));result=len(self.effects)
        if self.deduplicates:self.keys[operation_id]=(payload,result,None if self.lifetime is None else now+self.lifetime)
        if lose=='after':raise LostReply('reply unavailable')
        return result


def retry_cases():
    counts=[]
    for dedup,life,fault in [(False,None,'after'),(True,None,'after'),(True,1,'after'),(False,None,'before')]:
        s=ToyService(dedup,life)
        try:s.submit('op-1','add A',now=0,lose=fault)
        except LostReply:pass
        s.submit('op-1','add A',now=1);counts.append(len(s.effects))
    return counts


