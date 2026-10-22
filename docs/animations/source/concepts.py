"""Original educational animation data, adapted from the repositories' learning demos.

All numbers are deterministic toy calculations or explicitly assumed inputs.
Nothing here runs a model, discovers hardware, or contacts a service.
"""
import math

DURATION = 24
FPS = 12
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


def scenes(*entries):
    return [{'start':i*6,'end':(i+1)*6,'label':label,'caption':caption,'description':description}
            for i,(label,caption,description) in enumerate(entries)]


STORIES={
 'rehearsal-selection':{
  'title':'Learn today without losing yesterday',
  'subtitle':'Rehearsal and selection are different questions.',
  'repo':'neurofunctional-lab',
  'badge':'OFFLINE LINEAR TOY · NOT BENCHMARK EVIDENCE',
  'poster_alt':'Illustrative linear learner: B-only learning changes A prediction from one to about one-half; replay retains A. High current loss and greatest predicted loss increase select different stored examples.',
  'scenes':scenes(
   ('Compatible tasks','Shared weights can solve both tasks. New updates can still damage old predictions.',
    'The linear prediction is w1*x1+w2*x2. Task A has input (1,0), target 1; B has input (1,1), target 0. The shared solution (1,-1) fits both. Learning only A reaches approximately (1,0).'),
   ('Forgetting / rehearsal','Same model and later-update count. Replay changes access to old data.',
    'From the learned-A state, 120 updates containing B twice approach (0.5,-0.5). A predicts 0.5 and B predicts 0. Replacing one B presentation with stored A approaches (0.999,-0.998), retaining both. Each later condition has 240 example presentations, but replay stores one old example and uses fewer B presentations. Values come from the offline demo, not image benchmarks.'),
   ('Which example next?','Hard now and threatened next are different rankings. The virtual step is discarded.',
    'At weights (1,0), a hypothetical B step gives (0.75,-0.25). Current half-squared losses for old anchor, hard-now and threatened-next examples are 0, 0.5 and 0. Signed loss changes are +0.03125, -0.21875 and +0.125. High-loss selects hard-now; predicted-interference selects threatened-next. Uniform selection gives one-third to each. This is a scoring distinction, not a full MIR implementation or evidence that either selection improves a real model.'),
   ('Keep the claims separate','Buffer retention asks what remains. Selection asks what to rehearse from it.',
    'One toy demonstrates forgetting and possible protection; another illustrates different replay scores. Neither establishes a benchmark advantage, biological fidelity, an optimal selector or guaranteed protection. Read the repository\'s controlled studies separately.'))},
 'coordinate-readout':{
  'title':'Same distance. Different decisions.',
  'subtitle':'Translation, fixed readouts and projection are not interchangeable.',
  'repo':'representation-atlas','badge':'EXACT GEOMETRY ILLUSTRATION · NOT A MODEL EXPERIMENT',
  'poster_alt':'Translation preserves the distance between A and B but changes origin-based cosine. A fixed readout can change its decision; compensating its bias is a second operation. Projection merges differently labeled points.',
  'scenes':scenes(
   ('Translate both points','The shared offset cancels in distances, but not in directions from the origin.',
    'A=(1,0) and B=(0,1) both move by (2,2), ending at (3,2) and (2,3). Their Euclidean distance remains sqrt(2). Their cosine changes from 0 to 12/13, approximately 0.923. During the motion, distance and cosine are calculated from the current points.'),
   ('Hold the readout fixed','Moving an activation inside a fixed model is not just renaming coordinates.',
    'The fixed score x1-0.5 gives A=+0.5, B=-0.5 before translation and A=+2.5, B=+1.5 afterward. B crosses a zero threshold. Changing the bias to -2.5 restores +0.5 and -0.5, but this compensation is an additional operation.'),
   ('Discard a direction','Projection can erase the feature that carries the label.',
    'The points (1,1), labeled positive y, and (1,-1), labeled negative y, are projected onto x. Both become (1,0). A classifier seeing only that projected point cannot perfectly distinguish these labels. The projection is hand chosen, not fitted PCA.'),
   ('Ask what was preserved','Geometric invariance does not guarantee an unchanged downstream answer.',
    'Check pairwise geometry, downstream scores and task labels separately. These exact constructed examples are not results from a pretrained model and do not imply universal success or failure of component removal.'))},
 'content-explanation':{
  'title':'Change the wording. Keep the facts.',
  'subtitle':'Presentation agreement is not correctness or comprehension.',
  'repo':'adaptive-register','badge':'FIXED TEMPLATE ILLUSTRATION · NO MODEL OR READER EXPERIMENT',
  'poster_alt':'A ledger with Bay 3, 18:45 and printed ID feeds three fixed wording templates. A shorter sentence loses the facts; a wrong input remains wrong in every template. Literal checks do not establish truth or comprehension.',
  'scenes':scenes(
   ('Start with the ledger','The supplied facts are the content. The requested register controls presentation.',
    'The example ledger has place Bay 3, time 18:45 and requirement printed ID. Register means an explicitly requested presentation format, not an inference about a reader\'s ability or background.'),
   ('Three requested formats','Fixed templates retain the same three literals across all requested formats.',
    'Introductory: Go to Bay 3 at 18:45. Bring printed ID. Working: Check-in: Bay 3, 18:45. Required: printed ID. Technical: Location=Bay 3; time=18:45; requirement=printed ID. These are chosen templates with no learning or language model.'),
   ('Shorter can lose content','A simpler-looking sentence can omit information needed to act.',
    'Check in this evening. Bring your ID. This drops the exact place, time and printed-copy requirement. An exact-literal check catches those omissions here; it is not a semantic entailment test and can fail on paraphrases or negation.'),
   ('Consistent can be wrong','Preservation, truth and reader understanding require different checks.',
    'If the ledger incorrectly says Bay 9, all three templates repeat Bay 9. Agreement does not verify the original fact. This illustration establishes neither improved comprehension nor an auxiliary-head training benefit.'))},
 'capacity-latency':{
  'title':'More memory is not automatically more speed',
  'subtitle':'Separate an aggregate capacity floor from a sequential communication cost.',
  'repo':'consent-compute-mesh','badge':'CALCULATED ASSUMPTIONS · NOT DEVICE OR MODEL MEASUREMENTS',
  'poster_alt':'A hypothetical 70B model has 32.596 GiB ideal 4-bit weights and a 43.616 GiB illustrative runtime envelope. Assumed 12-GiB allocatable pools yield aggregate floors of three and four. Serial link delays grow with boundaries, not memory capacity.',
  'scenes':scenes(
   ('Count bytes first','Ideal weight packing is only one part of a runtime memory budget.',
    '70 billion parameters times 4 bits divided by 8 is 35 billion bytes: 35.00 GB or 32.596 GiB. The illustrative envelope adds 20 percent weight overhead, 2.5 GiB FP16 KV cache and 2 GiB workspace, totaling 43.616 GiB. KV dimensions are 80 layers, 8 KV heads, head dimension 128, context 8192 and batch one.'),
   ('Capacity lower bounds','At an assumed 12 GiB allocatable per pool: three for weights, four for the envelope.',
    'The ceiling of required bytes divided by assumed allocatable bytes gives an aggregate lower bound, not a placement guarantee. The pools must be simultaneously allocatable after reservations; no real free-memory measurements are used. Pool count does not automatically equal stage count.'),
   ('One dependent request','Each boundary adds latency plus serialization time before the next stage can proceed.',
    'For one 16-KiB boundary over hypothetical 20-Mbps payload bandwidth with 20-ms one-way latency, the transfer time is 20 ms plus 16384 divided by 2500000 seconds, or 26.5536 ms. Four sequential stages require three such crossings, totaling 79.6608 ms of communication alone. The animated marker illustrates dependencies, not real-time measured packet motion.'),
   ('Scale the boundaries','More serial crossings raise a communication-only floor, not an achieved token rate.',
    'For the declared one-token serial decoding graph with H sequential stages, Tcomm=(H-1)*(latency+payload/byte_rate). At four stages it is 0.0796608 seconds; the communication-only ceiling is 12.553 tokens per second. At sixteen stages it is 0.398304 seconds; the ceiling is 2.511 tokens per second. These exclude compute, queueing, protocol, return traffic, relay, contention and failures. Overlap across independent requests can alter throughput; it does not remove the dependency in this declared one-token serial graph. Speculative or multi-token decoding and different orchestration graphs need a different analysis; this is not a universal lower bound on all distributed inference.'))},
 'retry-idempotency':{
  'title':'A lost reply is not a failed write',
  'subtitle':'The service state and the caller\'s knowledge can diverge.',
  'repo':'research-workflow-atlas','badge':'SYNTHETIC CONTRACTS · NOT REAL API GUARANTEES',
  'poster_alt':'The same timeout can conceal either no commit or one commit. Retrying the same ID produces two effects without deduplication, one with a durable payload-bound key and two after key expiry. Unknown must remain unknown without reliable evidence.',
  'scenes':scenes(
   ('Write / lost reply','The service can commit successfully even when its reply never reaches the caller.',
    'A caller submits op-1, add A. The toy service commits one effect and loses the reply. The caller observes reply unavailable. Hidden effects are evaluator-only ground truth, not information contained in the timeout.'),
   ('Same observation','A timeout cannot distinguish a request that never committed from a lost post-commit reply.',
    'Two hidden histories produce the same caller observation: no initial commit with zero effects, or an initial commit with one effect. The caller sees reply unavailable in both. A missing or stale lookup is not automatically proof of noncommit.'),
   ('Retry the same ID','An ID is only a label unless the service provides a valid deduplication contract.',
    'One sequential retry after an initial commit yields two effects with ID only and no deduplication; one with a retained durable payload-bound key; and two when a key expires exactly at retry time. If the initial call never committed, a no-dedup retry yields one effect. Effects survive key expiry; cache hits do not renew lifetime.'),
   ('Resolve or preserve unknown','Use operation-specific evidence and the real retention contract before deciding to retry.',
    'The toy uses atomic calls, a single sequential caller and no late in-flight original. A live recovery decision also needs freshness, payload scope, expiry and whether the original may commit later. A reliable committed result may resolve uncertainty; otherwise preserve unknown. No real request or write occurs in this animation.'))}
}
