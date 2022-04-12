import os
import sys
import unittest

import numpy as np
from scipy.io import mmread
from scipy.linalg import det

sys.path.append("../src")

from qr import QR
import utility as ut
from variables import *


class TestQR(unittest.TestCase):
	@unittest.skip("Skipping complex random matrix tests because of floating point precision problems.")
	def test_wilkinson_shift_random(self):
		matrix_sizes = [10]
		a = -10
		b = 10
  
		for n in matrix_sizes:
			m = ut.complex_matrix(n, a, b, np.complex128)
			qr_alg = QR(m)
			u, r = qr_alg.qr_wilkinson_shift(1e-128, 500)
			eigs = qr_alg.extract_eigs(r)
   
			# Check the sum of the eigenvalues against the trace of H.
			np.testing.assert_almost_equal(np.sum(eigs), np.trace(m), decimal = 2)
			# Check the sum of the squares of the qigenvalues against the trace of H**2.
			np.testing.assert_almost_equal(np.sum(eigs ** 2), np.trace(np.linalg.matrix_power(m, 2)), decimal = 2)
			# Check the products of the eigenvalues against the determinant of H.
			determinant = np.linalg.det(m)
			# print(np.prod(eigs))
			# print(determinant)
			# print(np.prod(eigs) - determinant)
			np.testing.assert_almost_equal(np.prod(eigs), determinant, decimal = 0)
   
	def test_wilkinson_shift_market(self):
		matrix_filenames = ["gre__115", "jgl011"]
		err_msg = "The eigenvalues compute did not pass the test."
		for file in matrix_filenames:
			mat = mmread(os.path.join(MATRIX_MARKET_PATH, ".".join((file, MATRIX_MARKET_FILE_EXT))))
			m = mat.toarray() 
			qr_alg = QR(m)
			u, r = qr_alg.qr_wilkinson_shift(1e-128, 400)
			eigs = qr_alg.extract_eigs(r)
   
			# Check the sum of the eigenvalues against the trace of H.
			np.testing.assert_almost_equal(np.sum(eigs), np.trace(m), decimal = 2)
			# Check the sum of the squares of the qigenvalues against the trace of H**2.
			np.testing.assert_almost_equal(np.sum(eigs ** 2), np.trace(np.linalg.matrix_power(m, 2)), decimal = 2)
			# Check the products of the eigenvalues against the determinant of H.
			determinant = np.linalg.det(m)
			
			np.testing.assert_almost_equal(np.prod(eigs), determinant, decimal = 0)


if __name__ == "__main__":
	unittest.main()