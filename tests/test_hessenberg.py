import os
import sys
import unittest

import numpy as np
from scipy.io import mmread
from scipy.linalg import eig

# Add qr as an import module 
# to the PYTHONPATH.
sys.path.append(os.path.abspath("../src"))

import hessenberg as hg
from qr import QR
import utility as ut
from variables import *


class TestHessenberg(unittest.TestCase):
	def test_reflector(self):
		# Create random vectors.
		random_vectors_complex = ut.complex_matrix(5, 0, 10)
		random_vectors = np.random.default_rng().random(size = (5, 5), dtype = np.float32)
  
		for vec in random_vectors:
			# Get the Householder vector.
			t = hg.householder_reflector(vec)
	
			# Norm**2 of the Householder vector.
			t_norm_squared = t.T @ t
   
			# Form the Householder matrix.
			p = np.eye(vec.shape[0]) - (2 / t_norm_squared) * (np.outer(t, t.T)) 
   
			transformed_vec = p @ vec
	
			np.testing.assert_array_almost_equal(transformed_vec[1 :], np.zeros(vec.shape[0] - 1))
   
   
		for vec in random_vectors_complex:
			# Get the Householder vector.
			t = hg.householder_reflector(vec)
	
			# Norm**2 of the Householder vector.
			t_norm_squared = np.conj(t).T @ t
	
			# Form the Householder matrix.
			p = np.eye(vec.shape[0]) - (2 / t_norm_squared) * (np.outer(t, np.conj(t).T)) 
   
			transformed_vec = p @ vec
	
			np.testing.assert_array_almost_equal(transformed_vec[1 :], np.zeros(vec.shape[0] - 1))
			
	def test_hessenberg_transform_random(self):
		matrix_sizes = [10, 100, 1000]
		err_msg = "The eigenvalues computed from the hessenberg transform and the original matrix are not close enough."
  
		for n in matrix_sizes:
			a = -1e3 * np.random.default_rng().random(1) + 1.0
			b = 1e3 * np.random.default_rng().random(1) + 1.0
			m = ut.complex_matrix(n, a, b)
			hess = hg.hessenberg_transform(m, False)
			# Sort the eigevalues for comparison.
			eigs_hess = np.sort(eig(hess)[0])[::-1]
			eigs_original = np.sort(eig(m)[0])[::-1]
   
			np.testing.assert_allclose(actual = eigs_hess, desired = eigs_original, rtol = 0.0, atol = 1e-8, err_msg = err_msg)
   
	def test_hessenberg_transform_market(self):
		matrix_filenames = ["jgl011"]
		err_msg = "The eigenvalues computed from the hessenberg transform and the original matrix are not close enough."
		for file in matrix_filenames:
			mat = mmread(os.path.join(MATRIX_MARKET_PATH, ".".join((file, MATRIX_MARKET_FILE_EXT))))
			m = mat.toarray()
			hess = hg.hessenberg_transform(m, False)
			# Sort the eigevalues for comparison.
			eigs_hess = np.sort(eig(hess)[0])[::-1]
			eigs_original = np.sort(eig(m)[0])[::-1]
   
			np.testing.assert_allclose(actual = eigs_hess, desired = eigs_original, rtol = 0.0, atol = 1e-8, err_msg = err_msg)


if __name__ == "__main__":
    unittest.main()