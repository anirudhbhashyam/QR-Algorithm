"""
QR Decomposition
====================================
"""
import os
from typing import Union

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.linalg import eig, qr, schur, det

from hessenberg import *
from utility import *

path = "../test_matrices"
ext = "mtx.gz"

class QR:
	"""
 	A simple implementation of the QR algorithm and its variations.
	"""
	
	def __init__(self, M: np.ndarray):
		"""
		Initalises the QR class. The class stores a square matrix M in hessenberg form. 

		Parameters
		----------
		M:
			A square complex matrix.
   
		Raises
		------
			TypeError:
				If `M` is not of type `numpy ndarray`.
			AttributeError:
				If `M` is not square.
		"""
		if not isinstance(M, np.ndarray):
			raise TypeError("Input matrix must of type np.ndarray.")
	 
		if M.shape[0] != M.shape[1]:
			raise AttributeError(f"Matrix must be square given shape is {M.shape[0]} x {M.shape[1]}.")

		# Store matrix in Hessenberg form.
		self.H = hessenberg_transform(M, False)

	@staticmethod
	def givens_rotation(i: int, j: int, x: np.array, n: int) -> np.ndarray:
		"""
		Generates an `n` :math:`\\times` `n` special Givens Rotation matrix based on the parameters `i`, `j`, `x`, and `n`. The rotation matrix acts to reduce the `j`th component of the vector `x` to 0. For a Givens rotation matrix :math:`G`, vector :math:`u` and index :math:`j`
  
		.. math:: 
			G u = 
			\\left[
				\\begin{array}{c}
					u_1 \\\\
					\\vdots \\\\
					u_{j - 1} \\\\
					0 \\\\
					u_{j - 1} \\\\
					\\vdots \\\\
					u_{n} \\\\
				\\end{array}
			\\right]

		Parameters
		----------
		i:	
  			`i`th row.
		j:	
  			`j`th column.
		x: 	
  			Vector who's `j`th
			entry needs to be 
			reduced to 0.
		n: 	
  			Size of the returned
			Givens Matrix.

		Returns
		-------
		`numpy ndarray`:
			A Givens rotation matrix.
		"""

		givens_matrix = np.eye(n, dtype = np.complex256)
		reduced_x = x[i : i + 2]
  
		if reduced_x[1] == 0:
			c = 1
			s = 0
		else:
			if np.abs(reduced_x[1]) > np.abs(reduced_x[0]):
				r = -reduced_x[0] / reduced_x[1]
				s = 1 / np.sqrt(1 + np.power(r, 2))
				c = r * s
			else:
				r = -reduced_x[1] / reduced_x[0]
				c = 1 / np.sqrt(1 + np.power(r, 2))
				s = c * r
	 
		# Set the rotation elements.
		givens_matrix[i, i] = givens_matrix[j, j] = c
		givens_matrix[i, j] = s
		givens_matrix[j, i] = -s

		return givens_matrix

	@staticmethod 
	def givens_22(x: np.array):
		# print(f"x array type: {x.dtype}")
		g = np.zeros((2, 2), dtype = x.dtype)
	
		if x[1] == 0.0:
			c = 1.0
			s = 0.0
		elif x[0] == 0.0:
			c = 0.0
			s = sign(np.conj(x[1]))
		else:
			abs_x0_2 = x[0].real ** 2 + x[0].imag ** 2
			abs_x1_2 = x[1].real ** 2 + x[1].imag ** 2 
			denom = np.sqrt(abs_x0_2 + abs_x1_2)
			c = np.sqrt(abs_x0_2) / denom
			s = (sign(x[0]) * np.conj(x[1])) / denom
		
		g[0, 0] = c
		g[1, 1] = c
		g[0, 1] = s
		g[1, 0] = np.conj(-s)
		# print(f"{g = }")
		return g

	def qr_hessenberg(self, M: np.ndarray) -> (np.ndarray, np.ndarray):
		"""
		Performs one step of the :math:`QR` algorithm using a hessenberg matrix `H`. Procedurally generates the :math:`QR` decomposition of `H` exploiting the fact that `H` is in hessenberg form.

		Parameters
		----------
		H:	
  			An `n` :math:`\\times` `n` hessenberg 
			matrix.

		Returns
		-------
		`numpy ndarray`
			`Q`.
		`numpy ndarray`
			`RQ`.
		"""
		givens_matrices = list()
		r = M.copy()
		n = r.shape[0]
		q = np.eye(n, dtype = r.dtype)

		# Get QR decomposition using (g_n g_{n-1} ... g_1 H).
		for i in range(n - 1):
			# Retrieve the Givens Matrix.
			# g = self.givens_rotation(i, i + 1, r[:, i], n)
			g_ = self.givens_22(r[:, i][i : i + 2])
   
			# Since i and j for the Givens Matrix are consecutive
			# we get a 2x2 rotation matrix.
   
			# Store Q.
			q[ :, i : i + 2] = q[ :, i : i + 2] @ g_
			
			# Apply the Givens Matrices
			# from the left to H.
			r[i : i + 2, i :] = g_ @ r[i : i + 2, i :]
   
			# Store the Givens matrices to get H = RQ.
			givens_matrices.append(g_)
			

		# Get new H using (R g_1^* g_2^* ... g_n^*).
		for i in range(n - 1):
			# Apply the Givens Matrices to R
			# conjugated and transposed
			# from the right.
			r[: i + 2, i : i + 2] = r[: i + 2, i : i + 2] @ np.conj(givens_matrices[i]).T

		return q, r

	def qr_rayleigh_shift(self, eps: float, n_iter: int) -> (np.ndarray, np.ndarray):
		"""
		Performs the :math:`QR` algorithm employing the hessenberg method for the decomposition and utilises 
		the Rayleigh shift with deflation. Produces the Schur decomposition of :math:`H = URU^{*}`.
		
		Parameters
		----------
		eps:    
  			Tolerance to break 
			matrix self.H.
		n_iter: 
  			Number of iterations to
			perform.
		
		Returns
		-------
		`numpy ndarray`
			:math:`U`
		`numpy ndarray`
			:math:`R` 
		"""
		
		r = self.H.copy()
		n = r.shape[0]
		u = np.eye(n, dtype = r.dtype)
	
		# Iterate so that matrix is reduced
		# once off-diagonal elements are 
		# sufficiently small.
		for i in reversed(range(1, n)):
			k = 0
			while np.abs(r[i, i - 1]) > eps and k < n_iter:
				# Get shift for each iteration.
				sigma_k = r[i, i]

				# Shift H = H - sigma I.
				r -= sigma_k * np.eye(n)
		
				# Perform a step of the QR 
				# hessenberg method.
				q, r = self.qr_hessenberg(r)

				# Update H = H + sigma I.
				r += sigma_k * np.eye(n)
	
				# Form U.
				u = u @ q
				
				k += 1
	
		return u, r
	
	@staticmethod
	def wilkinson_shift(M: np.ndarray) -> float: 
		sigma = 0.5 * (M[0, 0] - M[1, 1])
		mu = M[1, 1] - sign(sigma) * (M[1, 0] ** 2)
		mu /= abs(sigma) + np.sqrt(sigma ** 2 + M[0, 0] ** 2)
		return mu

	def qr_wilkinson_shift(self, eps: float, n_iter: int) -> (np.ndarray, np.ndarray):
		"""
		Performs the QR algorithm  employing the hessenberg method for the decomposition and utilises 
		the Wilkinson shift. Produces the Schur decomposition of :math:`H = U^{*}RU`. 
		
		Parameters
		----------
		eps:    
  			Tolerance to break 
			matrix self.H.
   
		n_iter: 
  			Number of iterations to
			perform.
		
		Returns
		-------
		`numpy ndarray`
			:math:`U`
		`numpy ndarray`
			:math:`R`
		"""

		r = self.H.copy()
		n = r.shape[0]
		u = np.eye(n, dtype = r.dtype)
	
		# Iterate so that matrix is reduced
		# once off-diagonal elements are 
		# sufficiently small.
		for i in reversed(range(2, n)):
			k = 0
			while abs(r[i, i - 1]) > eps and k < n_iter:			 
    
				# if not r[i - 2:, i - 2:].size:
				# 	print(f"Matrix section {i = }")
				# 	with open("qr_output.txt", "a") as f:
				# 		f.write(f"iteration {i}:\n {pd.DataFrame(r).to_string(header = False, index = False)}\n")
     
				# Get shift for each iteration.
				sigma_k = self.wilkinson_shift(r[i - 2 :, i - 2 :])
				# print(f"{sigma_k = }")

				# Shift H.
				r -= sigma_k * np.eye(n, dtype = r.dtype)
		
				# Perform a step of the QR 
				# hessenberg method.
				q, r = self.qr_hessenberg(r)

				# Shift H back.
				r += sigma_k * np.eye(n, dtype = r.dtype)
    
				# Store the transformations.
				u = u @ q
				
				k += 1
	
		return u, r

	def double_shift(self, shift: complex):
		"""
		Performs an inefficient double shift for a real valued hessenberg matrix H 
		with complex eigenvalues. Calculates the real matrix :math:`M = H^2 - 2\\Re(\\sigma)H + |\\sigma|^2 I` and the double shifted :math:`H_2 = Q^{T} H Q`, where :math:`Q` is from the :math:`QR` decomposition of :math:`M`.
		
		Parameters
		----------
		shift:				
			Approximated complex eigenvalue.

		Returns
		-------
		`numpy ndarray`
			:math:`H_2`

		"""
		H = self.H.copy()
  
		# Calculate the real matrix M.
		M = H @ H - 2 * shift.real * H + (np.abs(shift) ** 2) * np.eye(H.shape[0])
		
		# QR factorisation of M.
		# Perform a step of the QR  
		# hessenberg method.
		_, q = self.qr_hessenberg(M)
		
		return q.T @ H @ q
		
	def francis_double_step(self, eps: float, n_iter: int):
		"""
		Performs an efficient version of the double shift algorithm to avoid complex
		airthmetic. Produces the Schur decomposition of :math:`H = URU^{*}`.
	 
	 	Parameters
		----------
		eps:    
  			Tolerance to break 
			matrix self.H.
   
		n_iter: 
  			Number of iterations to
			perform.
   
		Returns
		-------
		`numpy ndarray`
			:math:`U`
		`numpy ndarray`
			:math:`R`
  		"""
		H = self.H.copy()
		n = H.shape[0]
		active_size = n

		while active_size > 1:
			reduced_size = active_size - 2
			H_red = H[reduced_size : active_size, reduced_size : active_size]

			# Efficient real M = H^2 - coeff_1 H + coeff_2  I.
			coeff_1 = H_red[0, 0] - H_red[1, 1]
			coeff_2 = np.linalg.det(H_red)
   
			# First 3 elements of first column of M.
			x = H[0, 0] ** 2 + H[0, 1] * H[1, 0] - coeff_1 * H[0, 0] + coeff_2
			y = H[1, 0] * (H[0, 0] + H[1, 1] - coeff_1)
			z = H[1, 0] * H[2, 1]

			first_col_M_3 = np.array([x, y, z], dtype = np.float32)
   
			for k in range(active_size - 4):
				t = householder_reflector(first_col_M_3)
	
				# Norm ** 2 of the Householder vector.
				t_norm_squared = t.conj().T @ t
	
				p = np.eye(t.shape[0]) - 2 * (np.outer(t, t)) / t_norm_squared
	
				r = max((0, k))    
				H[k : k + 2, r : n] = p.T @ H[k : k + 2, r : n]

				r = min((k + 3, active_size))
				H[0 : r, k + 1 : k + 2] = H[0 : r, k + 1 : k + 2] @ p

				x = H[k + 2, k]
				y = H[k + 1, k]
	
				if k < active_size - 4:
					z = H[k + 3, k]

			t = householder_reflector(np.array([x, y]))
			# Norm ** 2 of the Householder vector.
			t_norm_squared = t.conj().T @ t
   
			p = np.eye(t.shape[0]) - 2 * (np.outer(t, t)) / t_norm_squared
   
			H[reduced_size: active_size, active_size, active_size - 3 : n] = p @ H[reduced_size: active_size, active_size, active_size - 3 : n]
   
			H[: active_size, active_size - 2 : active_size] = H[: active_size, active_size - 2 : active_size] @ p
   
   
			if np.abs(H[active_size, reduced_size]) < eps * (H[active_size, active_size] + H[reduced_size, reduced_size]):
				H[active_size, reduced_size] = 0
				active_size -= 1
				reduced_size -= active_size - 2
	
			elif np.abs(H[active_size - 2, reduced_size - 2]) < eps * (H[reduced_size - 2, reduced_size - 2] + H[reduced_size, reduced_size]):
				H[active_size - 2, reduced_size - 2] = 0 
				active_size -= 2
				reduced_size -= active_size - 2

		return H

def main():
	pd.options.display.max_columns = 200
	pd.set_option("display.width", 1000)
	pd.set_option("display.float_format", lambda x: f"{x:.6f}" )
	n = 100
	a = 0.0
	b = 1e3 * np.random.default_rng().random(1) + 1.0
	m = complex_matrix(n, a, b)
	max_element = max(np.max(m.real), np.max(m.imag))
	m /= max_element
	qr_alg = QR(m)
	# print(f"Original matrix:\n {pd.DataFrame(qr_alg.H)}")
 
	# -- TEST GIVENS ROTATION -- #
	# x = np.array([1.0 + 0.0j, 2.0 + 0.8j])
	# g_ = qr_alg.givens_22(x)
	# print(f"{g_ @ x}")
	# 
	
	# q, r = qr_alg.qr_hessenberg(qr_alg.H)
	# q_, r_ = qr(qr_alg.H)
	# print(f"q:\n {pd.DataFrame(q)}")
	# print(f"r:\n {pd.DataFrame(r)}")
	# print(f"q_:\n {pd.DataFrame(q_)}")
	# print(f"r_:\n {pd.DataFrame(r_)}")
 
	# -- TEST Rayleigh -- #
	# u, r = qr_alg.qr_rayleigh_shift(1e-12, 100)
	# t, _ = schur(qr_alg.H)
	# print(f"Rayleigh shift (r):\n {pd.DataFrame(r)}")
	# print(f"Eigs:\n {pd.DataFrame(np.sort(eig(qr_alg.H)[0])[::-1])}")
	# print(f"Schur form (scipy) r: {pd.DataFrame(t)}")
 
	# -- TEST Wilkison -- #
	u, r = qr_alg.qr_wilkinson_shift(1e-6, 100)
	# t, _ = schur(qr_alg.H)
	# print(f"Wilkison shift reconstruction (r):\n {pd.DataFrame(u.conj().T @ r @ u)}")
	# # print(f"Wilkinson shift (r):\n {pd.DataFrame(np.sort(np.diag(r).astype(np.complex128))[::-1])}")
	# print(f"Eigs:\n {pd.DataFrame(np.sort(eig(qr_alg.H)[0])[::-1])}")
	# print(f"Eigs dtype: {eig(qr_alg.H)[0].dtype}")
	# print(f"Schur form (scipy) r: {pd.DataFrame(t)}")
	# with open("output_qr.txt", "w") as f:
	# 	f.write(f"{pd.DataFrame(np.vstack([np.sort(eig(qr_alg.H)[0]), np.sort(np.diag(r).astype(np.complex128))]).T).to_string()}")

	print(f"Trace of H: {np.trace(qr_alg.H)}")
	print(f"Sum of eigenvalues: {np.trace(r)}")
	print(f"Trace of H**2: {np.trace(qr_alg.H @ qr_alg.H)}")
	print(f"Sum of eigenvalues**2: {np.trace(r @ r)}")
	# for i in range(n):
	# 	print(f"det(A - \u03BB I): {np.linalg.det(m.astype(np.complex128) - r.astype(np.complex128)[i, i] * np.eye(n))}")

	determinant = det(qr_alg.H.astype(np.complex128))
	# print(f"Determinant: {determinant}")
	np.testing.assert_allclose(np.prod(np.diag(r.astype(np.complex128))), determinant, atol = 0.00, rtol = 1.00)
	
	## -- Matrix Market -- ##
	# matrix_filenames = ["gre__115", "west0381"]
	# print("Testing the Wilkinson shift using matrices from the matrix market.")
	# err_msg = "The eigenvalues compute did not pass the test."
	# for file in matrix_filenames:
	# 	mat = mmread(os.path.join(path, ".".join((file, ext))))
	# 	m = mat.toarray() 
	# 	qr_alg = QR(m)
	# 	u, r = qr_alg.qr_wilkinson_shift(1e-6, 50)
	# 	# Check the sum of the eigenvalues against the trace of H.
	# 	np.testing.assert_allclose(np.trace(r), np.trace(qr_alg.H), rtol = 1e-6)
	# 	# Check the sum of the squares of the qigenvalues against the trace of H**2.
	# 	np.testing.assert_allclose(np.trace(r @ r), np.trace(qr_alg.H @ qr_alg.H), rtol = 1e-6)	
	# 	# Check the products of the eigenvalues against the determinant of H.
	# 	determinant = det(m)
	# 	np.testing.assert_allclose(np.prod(np.diag(r)), determinant, rtol = 1e-2)

if __name__ == "__main__":
	main()