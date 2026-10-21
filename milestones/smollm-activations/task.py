"""Exactly verifiable, answer-identity-balanced arithmetic contrast pairs."""
def examples():
 rows=[]
 for a in range(10):
  for b in range(10):
   correct=(a+b)%10;wrong=(correct+3)%10
   rows.append({'id':f'{a}-{b}','prompt':f'Question: What is the last digit of {a} + {b}?\nAnswer:','correct':str(correct),'incorrect':str(wrong)})
 return rows
