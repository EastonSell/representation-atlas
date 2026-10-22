"""Outcome-blind lexical selection. This module never imports a model class."""
import hashlib
PROMPT='Repeat the word.\nWord: {word}\nRepeated word:'
POLICY='ascending sha256(UTF-8 copy-patch-v1|word); first 64; adjacent pairs; both orientations'
class InsufficientCandidates(ValueError):
 def __init__(self,ledger): self.ledger=ledger;super().__init__('Fewer than 64 eligible candidates; retained ledger; no replacements permitted')
def encode(tok,text):return list(tok(text,add_special_tokens=False)['input_ids'])
def select(tok,words,excluded):
 seen=set();eligible=[];rejected=[];reference=None
 for index,word in enumerate(words):
  reason=None;entry=None
  if not isinstance(word,str) or not word.isascii() or not word.isalpha() or word.lower()!=word:reason='not_lowercase_ascii_alphabetic'
  elif word in seen:reason='duplicate'
  elif word in excluded:reason='historical_exclusion'
  else:
   text=PROMPT.format(word=word); answer=encode(tok,' '+word)
   tokenized=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=list(tokenized['input_ids'])
   start=len('Repeat the word.\nWord: ');end=start+len(word)
   positions=[i for i,(a,b) in enumerate(tokenized['offset_mapping']) if b>start and a<end]
   if len(answer)!=1:reason='answer_not_single_token'
   elif encode(tok,text+' '+word)!=ids+answer:reason='answer_not_append_only'
   elif len(positions)!=1:reason='source_not_single_token'
   elif len(ids)>32:reason='prefix_exceeds_32'
   else:
    p=positions[0];a,b=tokenized['offset_mapping'][p]
    if a>start or b<end:reason='source_span_not_covered'
    elif not 0<p<len(ids)-1:reason='invalid_positions'
    elif reference and (len(ids)!=len(reference[0]) or p!=reference[1] or any(x!=y for i,(x,y) in enumerate(zip(ids,reference[0])) if i!=p)):reason='reference_alignment_mismatch'
    else:
     if reference is None:reference=(ids,p)
     entry={'word':word,'ids':ids,'answer':answer[0],'p':p}
  if isinstance(word,str):seen.add(word)
  if reason:rejected.append({'candidate_index':index,'word':word,'reason':reason})
  else:eligible.append(entry)
 eligible.sort(key=lambda x:hashlib.sha256(('copy-patch-v1|'+x['word']).encode()).hexdigest())
 result={'schema_version':1,'candidate_count':len(words),'eligible_count':len(eligible),'rejected':rejected,'eligible_words':[x['word'] for x in eligible],'selection_policy':POLICY,'rows':[]}
 if len(eligible)<64:raise InsufficientCandidates(result)
 chosen=eligible[:64]
 for i,e in enumerate(chosen):
  d=chosen[i^1]
  if e['answer']==d['answer']:raise ValueError('Paired answer collision: abort, no replacement')
  if encode(tok,PROMPT.format(word=e['word'])+' '+d['word'])!=e['ids']+[d['answer']]:raise ValueError('Cross-prefix append mismatch: abort, no replacement')
  result['rows'].append(dict(row_id=i,pair_id=i//2,orientation=i%2,recipient_word=e['word'],donor_word=d['word'],recipient_token_id=e['answer'],donor_token_id=d['answer'],recipient_prefix_ids=e['ids'],donor_row_id=i^1,source_position=e['p'],negative_position=e['p']-1,final_position=len(e['ids'])-1,prefix_length=len(e['ids'])))
 validate(result['rows']);return result
def validate(rows):
 def need(x):
  if not x:raise ValueError('Dataset invariant violated')
 need([r['row_id'] for r in rows]==list(range(64)));need(len(rows)==64);need({r['row_id'] for r in rows}==set(range(64)));by={r['row_id']:r for r in rows}
 need(len({r['recipient_word'] for r in rows})==64);need({r['pair_id'] for r in rows}==set(range(32)))
 for r in rows:
  i=r['row_id'];d=by[r['donor_row_id']];p=r['source_position'];ids=r['recipient_prefix_ids']
  need(all(type(r[k]) is int and 0<=r[k]<49152 for k in ('recipient_token_id','donor_token_id')))
  need(r['pair_id']==i//2 and r['orientation']==i%2 and r['donor_row_id']==i^1)
  need(r['recipient_word']==d['donor_word'] and r['donor_word']==d['recipient_word'])
  need(r['recipient_token_id']==d['donor_token_id'] and r['donor_token_id']==d['recipient_token_id'] and r['recipient_token_id']!=r['donor_token_id'])
  need(r['prefix_length']==len(ids)<=32 and r['final_position']==len(ids)-1 and r['negative_position']==p-1 and 0<p<r['final_position'])
  need(all(isinstance(t,int) and 0<=t<49152 for t in ids));need(len(ids)==len(d['recipient_prefix_ids']) and p==d['source_position'])
  need([j for j,(a,b) in enumerate(zip(ids,d['recipient_prefix_ids'])) if a!=b]==[p])
