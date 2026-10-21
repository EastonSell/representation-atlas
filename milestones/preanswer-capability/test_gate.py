import unittest
from pathlib import Path
class GateTests(unittest.TestCase):
 def test_implementation_exists(self):
  self.assertTrue(Path('gate.py').exists(), 'Capability implementation missing')
 def test_data_balance(self):
  if not Path('gate.py').exists(): self.skipTest('implementation absent')
  from gate import examples
  rows=examples(); self.assertEqual(len(rows),24)
  for i in range(0,24,2):
   a,b=rows[i:i+2]; self.assertEqual(a['pair'],b['pair']);self.assertNotEqual(a['answer'],b['answer'])
   self.assertEqual(a['prompt'].replace('of '+a['target']+' is '+a['answer'], 'TARGET VALUE'),b['prompt'].replace('of '+b['target']+' is '+b['answer'], 'TARGET VALUE'))
 def test_parser(self):
  if not Path('gate.py').exists(): self.skipTest('implementation absent')
  from gate import exact
  self.assertTrue(exact(' Red. ', 'red', False)); self.assertFalse(exact('red or blue','red',False));self.assertFalse(exact('red','red',True))
if __name__=='__main__':unittest.main()
