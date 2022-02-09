import os
import sys
import unittest

import numpy as np
from scipy.io import mmread
from scipy.linalg import hessenberg, eig, det


# Add qr as an import module 
# to the PYTHONPATH.
sys.path.append("../qr")

from hessenberg import *
from qr import QR
import utility as ut


# Path specification for test matrices 
# from matrix market. scipy.io.mmread
# is used to read gunzipped matrix files.
path = "../test_matrices"
ext = "mtx.gz"

class TestQR(unittest.TestCase):
	def test_wilkinson_shift_random(self):
		print("Testing the Wilkinson shift using random matrices.")
		matrix_sizes = [10, 100]
		a = 0.0
		b = 1e3 * np.random.default_rng().random(1) + 1.0
  
		for n in matrix_sizes:
			m = ut.complex_matrix(n, a, b)
			max_element = np.max(m.real) if np.max(m.real) >= np.max(m.imag) else np.max(m.imag)
			m /= max_element
			qr_alg = QR(m)
			u, r = qr_alg.qr_wilkinson_shift(1e-64, 500)
			eigs = qr_alg.extract_eigs(r)
   
			# Check the sum of the eigenvalues against the trace of H.
			np.testing.assert_allclose(np.trace(r), np.sum(eigs), rtol = 1e-6)
			# Check the sum of the squares of the qigenvalues against the trace of H**2.
			np.testing.assert_allclose(np.trace(r @ r), np.sum(eigs ** 2), rtol = 1e-6)
			# Check the products of the eigenvalues against the determinant of H.
			determinant = det(qr_alg.H.astype(np.complex128))
			np.testing.assert_allclose(np.prod(eigs), determinant, rtol = 1.0, atol = 0.0)
   
	def test_wilkinson_shift_market(self):
		matrix_filenames = ["gre__115", "west0381"]
		print("Testing the Wilkinson shift using matrices from the matrix market.")
		err_msg = "The eigenvalues compute did not pass the test."
		for file in matrix_filenames:
			mat = mmread(os.path.join(path, ".".join((file, ext))))
			m = mat.toarray() 
			qr_alg = QR(m)
			u, r = qr_alg.qr_wilkinson_shift(1e-64, 500)
			eigs = qr_alg.extract_eigs(r)
			
			# Check the sum of the eigenvalues against the trace of H.
			np.testing.assert_allclose(np.trace(r), np.sum(eigs), rtol = 1e-6)
			# Check the sum of the squares of the qigenvalues against the trace of H**2.
			np.testing.assert_allclose(np.trace(r @ r), np.sum(eigs ** 2), rtol = 1e-6)
			# Check the products of the eigenvalues against the determinant of H.
			determinant = det(qr_alg.H)
			np.testing.assert_allclose(np.prod(eigs), determinant, rtol = 1.0, atol = 0.0)


if __name__ == "__main__":
	unittest.main()