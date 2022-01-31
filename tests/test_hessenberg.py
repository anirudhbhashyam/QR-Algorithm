import os
import sys
import unittest

import numpy as np
from scipy.io import mmread
from scipy.linalg import hessenberg, eig

# Add qr as an import module 
# to the PYTHONPATH.
sys.path.append("../qr")

from hessenberg import *
from qr import QR

# Path specification for test matrices 
# from matrix market. scipy.io.mmread
# is used to read gunzipped matrix files.
path = "../test_matrices"
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
			
	def test_hessenberg_transform_random(self):
		matrix_sizes = [10, 100, 1000]
		err_msg = "The eigenvalues computed from the hessenberg transform (via scipy and via the implemented algorithm) are not close enough."
  
		for n in matrix_sizes:
			a = 0.0
			b = 1e3 * np.random.default_rng().random(1) + 1.0
			m = complex_matrix(n, a, b)
			hess = hessenberg_transform(m, False)
			hess_from_scipy = hessenberg(m)
			# Sort the eigevalues for comparison.
			eigs = np.sort(eig(hess)[0])
			eigs_scipy = np.sort(eig(hess_from_scipy)[0])
			np.testing.assert_allclose(actual = eigs, desired = eigs_scipy, rtol = 1e-6, err_msg = err_msg)
   
	def test_hessenberg_transform_market(self):
		matrix_filenames = ["gre__115", "west0381"]
		err_msg = "The eigenvalues computed from the hessenberg transform (via scipy and via the implemented algorithm) are not close enough."
		for file in matrix_filenames:
			mat = mmread(os.path.join(path, "".join((file, ext))))
			m = mat.toarray()
			hess = hessenberg_transform(m, False)
			hess_from_scipy = hessenberg(m) 
			# Sort the eigevalues for comparison.
			eigs = np.sort(eig(hess)[0])
			eigs_scipy = np.sort(eig(hess_from_scipy)[0])
			np.testing.assert_allclose(actual = eigs, desired = eigs_scipy, rtol = 1e-6, err_msg = err_msg)


if __name__ == "__main__":
    unittest.main()