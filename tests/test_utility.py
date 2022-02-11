import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath("../src"))

import utility as ut

class TestUtility(unittest.TestCase):
	def test_complex_matrix(self):
		a = 5
		b = 10
		n = np.random.default_rng().integers(low = 1, high = 10)
		self.assertEqual(ut.complex_matrix(n, a, b).shape[0], n)
		self.assertEqual(ut.complex_matrix(n, a, b, np.complex128).dtype, np.complex128)
  
	def test_sign(self):
		z = 3 + 4j
		a = 0
		b = -3.4
		self.assertEqual(ut.sign(z), 3/5 + (4/5) * 1j)
		self.assertEqual(ut.sign(a), 1)
		self.assertEqual(ut.sign(b), -1)
  
	def test_closeness(self):
		with self.assertRaises(ValueError) as ctx:
			ut.closeness([-1], [-2 + 6j, 0], 1e-6)

		self.assertTrue("Length of input arrays do not match" in str(ctx.exception))
			
  
		a = [1.00, 2.00, 3.00]
		b = [1.01, 2.02, 3.00]
		self.assertTrue(ut.closeness(a, b, 1e-1)[0])
		self.assertFalse(ut.closeness(a, b, 1e-3)[0])

		c = [-1.8 + 2.7j, 3.1890 + 4.2j]
		d = [-1.8 + 2.734j, 3.1 + 4.2j]
  
		self.assertTrue(ut.closeness(c, d, 1e-1)[0])
		self.assertFalse(ut.closeness(c, d, 1e-2)[0])

if __name__ == '__main__':
    unittest.main()
