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

class TestQR(unittest.TestCase):
	def test_2x2_eigs(self):
		err_msg = "The eigenvalues computed from the implemented method and through scipy are not close enough."
		for _ in range(4):
			a = 0.0
			b = 1e3 * np.random.default_rng().random(1) + 1.0
			m = complex_matrix(2, a, b)
			qr_alg = QR(m)
			eigs = np.sort(qr_alg.eig_22(m))
			eigs_scipy = np.sort(eig(m)[0])
			np.testing.assert_allclose(actual = eigs, desired = eigs_scipy, rtol = 1e-6, err_msg = err_msg)

		for _ in range(4):
			a = 0.0
			b = 1e3 * np.random.default_rng().random(1) + 1.0
			m = complex_matrix(2, a, b).real
			qr_alg = QR(m)
			eigs = np.sort(qr_alg.eig_22(m))
			eigs_scipy = np.sort(eig(m)[0])
			np.testing.assert_allclose(actual = eigs, desired = eigs_scipy, rtol = 1e-6, err_msg = err_msg)


	
			
		
if __name__ == "__main__":
	unittest.main()