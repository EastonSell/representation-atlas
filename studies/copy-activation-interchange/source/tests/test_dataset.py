import unittest, copy
from patchstudy import dataset
class Tokenizer:
 def __init__(self): self.v={};self.rev={}
 def __call__(self,s,add_special_tokens=False,return_offsets_mapping=False):
  import re
  parts=list(re.finditer(r' ?[a-zA-Z]+|[^a-zA-Z]',s)); ids=[]
  for m in parts:
   x=m.group(); self.v.setdefault(x,len(self.v)+1);ids.append(self.v[x])
  d={'input_ids':ids}
  if return_offsets_mapping:d['offset_mapping']=[m.span() for m in parts]
  return d
class DataTests(unittest.TestCase):
 def test_filter_preserves_rejections_and_selection(self):
  words=['w'+chr(97+i//26)+chr(97+i%26) for i in range(70)]
  result=dataset.select(Tokenizer(),words+['bad1',words[0],'old'],['old'])
  self.assertEqual(len(result['rows']),64);self.assertEqual(len(result['rejected']),3)
  self.assertEqual(result['eligible_count'],70); dataset.validate(result['rows'])
 def test_small_pool_stops(self):
  with self.assertRaises(dataset.InsufficientCandidates) as c:dataset.select(Tokenizer(),['air'],[])
  self.assertEqual(c.exception.ledger['eligible_count'],1)
 def test_tamper_fails(self):
  words=['w'+chr(97+i//26)+chr(97+i%26) for i in range(64)]
  rows=dataset.select(Tokenizer(),words,[])['rows'];rows[0]['donor_token_id']=rows[0]['recipient_token_id']
  with self.assertRaises(ValueError):dataset.validate(rows)
 def test_negative_answer_and_reordered_rows_rejected(self):
  words=['w'+chr(97+i//26)+chr(97+i%26) for i in range(64)]
  rows=dataset.select(Tokenizer(),words,[])['rows']
  with self.assertRaises(ValueError):dataset.validate(rows[::-1])
  rows[0]['recipient_token_id']=-1;rows[1]['donor_token_id']=-1
  with self.assertRaises(ValueError):dataset.validate(rows)
