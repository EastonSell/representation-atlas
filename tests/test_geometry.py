import unittest
import numpy as np
from geometry import *
class GeometryTests(unittest.TestCase):
 def setUp(self):
  self.rng=np.random.default_rng(7);self.x=self.rng.normal(size=(30,8))
 def test_centering_train_only(self):
  state=fit_transform(self.x);np.testing.assert_allclose(state[0],self.x.mean(0))
  test=self.x+100;self.assertGreater(transform(test,state).mean(),90)
 def test_translation_distances(self):
  z=transform(self.x,fit_transform(self.x));np.testing.assert_allclose(np.linalg.norm(self.x-self.x[0],axis=1),np.linalg.norm(z-z[0],axis=1))
 def test_pairing_mean_invariant(self):
  y=self.rng.normal(size=self.x.shape);np.testing.assert_allclose(direction(self.x,y),direction(self.x,y[::-1]),atol=1e-14)
 def test_removed_components(self):
  state=fit_transform(self.x,2);np.testing.assert_allclose(transform(self.x,state)@state[1].T,0,atol=1e-12)
 def test_rotation_cka(self):
  q=np.linalg.qr(self.rng.normal(size=(8,8)))[0];self.assertAlmostEqual(cka(self.x,self.x@q),1)
 def test_invalid(self):
  for x in [[],[[float('nan')]], [1,2]]:
   with self.assertRaises(ValueError): matrix(x)
  with self.assertRaises(ValueError): fit_transform(self.x,-1)
  with self.assertRaises(ValueError): cka(np.ones((3,2)),np.ones((3,2)))
if __name__=='__main__':unittest.main()
