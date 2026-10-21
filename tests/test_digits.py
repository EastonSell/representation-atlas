import unittest
import numpy as np
from types import SimpleNamespace
from run_digits import hidden,centroid_predict
class DigitGeometryTests(unittest.TestCase):
 def test_relu_activation(self):
  m=SimpleNamespace(coefs_=[np.eye(2)],intercepts_=[np.zeros(2)])
  np.testing.assert_array_equal(hidden(m,np.array([[-1,2]])),[[0,2]])
 def test_centroid_translation(self):
  x=np.array([[0.,0],[1,0],[10,10],[11,10]]);y=np.array([0,0,1,1]);t=np.array([[2.,1],[8,9]])
  np.testing.assert_array_equal(centroid_predict(x,y,t),centroid_predict(x+40,y,t+40))
if __name__=='__main__':unittest.main()
