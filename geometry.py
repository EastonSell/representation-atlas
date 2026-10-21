"""Train-only geometry controls; all arrays are samples by features."""
import numpy as np

def matrix(x):
    x=np.asarray(x,dtype=float)
    if x.ndim != 2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite samples-by-features matrix')
    return x

def fit_transform(train, remove=0):
    train=matrix(train)
    if not isinstance(remove,int) or not 0<=remove<min(train.shape):
        raise ValueError('remove must be an integer below matrix rank bound')
    mean=train.mean(axis=0)
    _,_,vt=np.linalg.svd(train-mean,full_matrices=False)
    return mean,vt[:remove]

def transform(x,state):
    mean,pcs=state; z=matrix(x)-mean
    return z-(z@pcs.T)@pcs

def direction(positive,negative):
    positive,negative=matrix(positive),matrix(negative)
    if positive.shape!=negative.shape: raise ValueError('Matched shapes required')
    return (positive-negative).mean(axis=0)

def ranking(positive,negative,axis):
    scores=(matrix(positive)-matrix(negative))@axis
    return float(np.mean((scores>0)+0.5*(scores==0)))

def cosine(a,b):
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide(np.sum(a*b,axis=1),den,out=np.zeros_like(den),where=den>0)

def cka(a,b):
    a,b=matrix(a),matrix(b)
    if len(a)!=len(b): raise ValueError('Aligned samples required')
    a=a-a.mean(0);b=b-b.mean(0)
    denominator=np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)
    if denominator==0: raise ValueError('CKA undefined for constant representation')
    return float(np.linalg.norm(a.T@b)**2/denominator)

def paired(rng,n,d,regime):
    if regime not in ('low_signal_variance','high_signal_variance'): raise ValueError(regime)
    topic=rng.normal(size=(n,d)); topic[:,0]=0
    topic[:,1]*=4
    strength=0.6 if regime=='low_signal_variance' else 12.0
    signal=np.zeros(d);signal[0]=strength
    offset=np.full(d,5.0)
    return (topic+signal+rng.normal(scale=1,size=(n,d))+offset,
            topic-signal+rng.normal(scale=1,size=(n,d))+offset)
