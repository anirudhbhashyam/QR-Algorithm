"""
QR Decomposition
====================================
"""
import numpy as np
import pandas as pd
from scipy.linalg import eig

# from hessenberg import *


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
  			ith row.
		j:	
  			jth column.
		x: 	
  			Vector who's jth
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

		givens_matrix = np.eye(n, dtype = np.complex128)

		# Cos component of the matrix.
		c = x[i] / np.sqrt(np.power(x[i], 2) + np.power(x[j], 2))

		# Sin component of the matrix.
		s = -x[j] / np.sqrt(np.power(x[i], 2) + np.power(x[j], 2))

		# Set the rotation elements.
		givens_matrix[i, i] = givens_matrix[j, j] = c
		givens_matrix[i, j] = s
		givens_matrix[j, i] = -s

		return givens_matrix

	@staticmethod
	def eig_22(M: np.ndarray) -> (float, float):
		"""
		Approximates the eigenvalues  of the 2 :math:`\\times` 2 complex matrix `M`.

		Parameters
		----------
		M:	
  			A 2 x 2 complex matrix. 

		Returns
		-------
		`tuple`
			Eigenvalues of `M`.
		"""
		if M.shape[0] != 2 or M.shape[1] != 2:
			raise ValueError(f"Provided matrix should have shape 2 x 2 but has shape {M.shape[0]} x {M.shape[1]}.")

		a = M[0, 0]
		b = M[0, 1]
		c = M[1, 0]
		d = M[1, 1]
		t_1 = 0.5 * (a + d - np.sqrt(np.power(a, 2) + np.power(d, 2) + 4 * b * c - 2 * a * d))
		t_2 = 0.5 * (a + d + np.sqrt(np.power(a, 2) + np.power(d, 2) + 4 * b * c - 2 * a * d))
		return t_1, t_2

	def qr_hessenberg(self, M: np.ndarray) -> (np.ndarray, np.ndarray):
		"""
		Performs one step of the :math:`QR` algorithm using a hessenberg matrix `H`. Procedurally generates the :math:`QR` decomposition of `H` exploiting the fact that `H` is in hessenberg form.

		Parameters
		----------
		H:	
  			An n x n hessenberg 
			matrix.

		Returns
		-------
		`numpy ndarray`
			`RQ`.
		`numpy ndarray`
			`Q`.
		"""
		givens_matrices = list()
		r = M.copy()
		n = r.shape[0]
		q = np.eye(n)

		# Get QR decomposition using (g_n^* g_{n-1}^* ... g_1^* H).
		for i in range(n - 1):
			# Retrieve the Givens Matrix.
			g = self.givens_rotation(i, i + 1, r[:, i], n)

			# Store Q.
			q = q @ g

			givens_matrices.append(g)
			
			# Apply the Givens Matrices
			# (conjugated and transposed)
			# from the left.
			r = g.conj().T @ r

		# Get new H using (R g_1 g_2 ... g_n).
		for i in range(n - 1):
			# Apply the Givens Matrices to R 
			# from the right.
			r = r @ givens_matrices[i] 

		return r, q

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
		u = np.eye(n)
	
		# Iterate so that matrix is reduced
		# once off-diagonal elements are 
		# sufficiently small.
		for i in reversed(range(2, n)):
			k = 0
			while np.abs(r[i, i - 1]) > eps and k < n_iter:
				# Get shift for each iteration.
				sigma_k = r[i, i]

				# Shift H.
				r -= sigma_k * np.eye(n)
		
				# Perform a step of the QR 
				# hessenberg method.
				r, q = self.qr_hessenberg(r)

				# Update H and U
				# H = H + sigma I		.
				r += sigma_k * np.eye(n)
				u = u @ q
				
				k += 1
	
		return u, r

	def qr_wilkinson_shift(self, eps: float, n_iter: int) -> (np.ndarray, np.ndarray):
		"""
		Performs the QR algorithm  employing the hessenberg method for the decomposition and utilises 
		the Wilkinson shift. Produces the Schur decomposition of :math:`H = URU^{*}`. 
		
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
		u = np.eye(n)
	
		# Iterate so that matrix is reduced
		# once off-diagonal elements are 
		# sufficiently small.
		for i in reversed(range(2, n)):
			k = 0
			while np.abs(r[i, i - 1]) > eps and k < n_iter:
				
				# Get shift for each iteration.
				sigma_k = self.eig_22(r[i - 2 : i, i - 2 : i])[0]
				# print(f"{sigma_k = }")

				# Shift H.
				r -= sigma_k * np.eye(n)
		
				# Perform a step of the QR 
				# hessenberg method.
				r, q = self.qr_hessenberg(r)

				# Update H and U.
				r += sigma_k * np.eye(n)
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
	n = 8
	a = 0.0
	b = 10.0
	M = complex_matrix(n, a, b)
	
	pd.options.display.max_columns = 200
	pd.set_option("display.width", 1000)
	# pd.set_option("display.float_format", lambda x: f"{x:.3f}" )
 
	print(f"Original matrix:\n {pd.DataFrame(M)}")
	qr_alg = QR(M)
	a = 0.0
	b = 1e3 * np.random.default_rng().random(1) + 1.0
	m = complex_matrix(2, a, b)
	qr_alg = QR(m)
	eigs = np.sort(qr_alg.eig_22(m))
	eigs_scipy = np.sort(eig(m)[0])
	print(f"Eigs:\n {pd.DataFrame(eigs)}")
	print(f"Eigs (scipy):\n {pd.DataFrame(eigs_scipy)}")
	np.testing.assert_allclose(actual = eigs, desired = eigs_scipy, rtol = 1e-6)
	# print(f"Hessenberged:\n {pd.DataFrame(qr_alg.H)}")
	# u, r = qr_alg.francis_double_step(1e-12, 30)
	# print(f"R:\n {pd.DataFrame(r).round(decimals = 2)}")
	# print(f"U:\n {pd.DataFrame(u).round(decimals = 2)}")
	# print(f"R:\n {pd.DataFrame(q)}")
	# u, r = qr_alg.qr_rayleigh_shift(1e-12, 20)
	# pd.set_option('display.max_columns', None)
	# print(pd.DataFrame(u))
	# print(pd.DataFrame(r))
	
if __name__ == "__main__":
	main()