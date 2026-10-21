import unittest
from collections import Counter
from task import examples
class TaskTests(unittest.TestCase):
 def test_verifiable_balanced_pairs(self):
  rows=examples();self.assertEqual(len(rows),100);self.assertEqual(len({r['id'] for r in rows}),100)
  self.assertEqual(Counter(r['correct'] for r in rows),Counter(r['incorrect'] for r in rows))
  for r in rows:
   a,b=map(int,r['id'].split('-'));self.assertEqual(int(r['correct']),(a+b)%10);self.assertNotEqual(r['correct'],r['incorrect'])
if __name__=='__main__':unittest.main()
