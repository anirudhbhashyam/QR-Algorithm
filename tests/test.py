import unittest
import numpy as np
import sys
import os

from scipy.io import mmread
from scipy.linalg import hessenberg

sys.path.append("../qr")

from qr import *



path = "../test_matrices"
files = ["blckhole"]
ext = ".mtx.gz"

class TestHessenberg(unittest.TestCase):
	def test_reflector(self):
		# Create random vectors.
		random_vectors = np.random.default_rng().random(size = (5, 5), dtype = np.float32)
  
		for vec in random_vectors:
			# Get the Householder vector.
			t = householder_reflector(vec)
	
			# Norm**2 of the Householder vector.
			t_norm_squared = t.conj().T @ t
	
			# Form the Householder matrix.
			p = np.eye(vec.shape[0]) - 2 * (np.outer(t, t)) / t_norm_squared
   
			transformed_vec = p @ vec
	
			np.testing.assert_array_almost_equal(transformed_vec[1 : ], np.zeros(vec.shape[0] - 1))
			
	def test_hessenberg_transform(self):
		for file in files:
			mat = mmread(os.path.join(path, "".join((file, ext))))
			m = mat.toarray()
			
			np.testing.assert_allclose(hessenberg_transform(mat)[0], hessenberg(mat), rtol = 1e-6)
  
if __name__ == "__main__":
    unittest.main()