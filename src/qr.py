"""
QR Decomposition
================
"""
import os
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.linalg import qr

import hessenberg as hg
import utility as ut
from variables import *

class QR:
	"""
 	A simple implementation of the QR algorithm and its variations.
	"""
	
	def __init__(self, M: np.ndarray):
		"""
		Initalises the QR class. The class stores a square matrix `M` in hessenberg form. 

		Parameters
		----------
		M:
			A square complex matrix.
   
		Raises
		------
			TypeError:
				If `M` is not of type `numpy ndarray`.
			ValueError:
				If `M` is not square.
		"""
		if not isinstance(M, np.ndarray):
			raise TypeError("Input matrix must of type np.ndarray.")
	 
		if M.shape[0] != M.shape[1]:
			raise ValueError(f"Matrix must be square but given shape is {M.shape}.")

		# Store matrix in Hessenberg form.
		self.H = hg.hessenberg_transform(M, False)
	

	@staticmethod
	def givens_rotation(i: int, j: int, x: np.array, n: int) -> np.ndarray:
		"""
		Generates an `n` :math:`\\times` `n` special Givens Rotation matrix based on the parameters `i`, `j`, `x`, and `n`. The rotation matrix acts to reduce the `j` th component of the vector `x` to 0. For a Givens rotation matrix :math:`G`, vector :math:`u` and index :math:`j`
  
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
		``numpy ndarray``:
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
				c = 1 / np.sqrt(1 + r ** 2)
				s = c * r
	 
		# Set the rotation elements.
		givens_matrix[i, i] = givens_matrix[j, j] = c
		givens_matrix[i, j] = s
		givens_matrix[j, i] = -s

		return givens_matrix

	@staticmethod 
	def givens_22(x: np.array) -> np.ndarray:
		"""
		Generates an :math:`2 \\times 2` special Givens Rotation matrix based on the on the :math:`2 \\times 1` vector `x`. Function is useful for producing Givens matrices stabily and efficiently. For a Givens rotation matrix :math:`G`, vector :math:`u`.
		
		.. math::
  
			G u = 
   			\\left[
				\\begin{array}{c}
					u_1 \\\\
					0 \\\\
				\\end{array}
			\\right]
   		
		Parameters
		----------
		x: 	
			:math:`2 \\times 2` vector who's 2nd entry needs to be reduced to 0.

		Returns
		-------
		``numpy ndarray``:
			A :math:`2 \\times 2` Givens rotation matrix.
		"""
		g = np.zeros((2, 2), dtype = x.dtype)
	
		if x[1] == 0.0:
			c = 1.0
			s = 0.0
		elif x[0] == 0.0:
			c = 0.0
			s = ut.sign(np.conj(x[1]))
		else:
			abs_x0_2 = x[0].real ** 2 + x[0].imag ** 2
			abs_x1_2 = x[1].real ** 2 + x[1].imag ** 2
			denom = np.sqrt(abs_x0_2 + abs_x1_2)
			c = np.sqrt(abs_x0_2) / denom
			s = (ut.sign(x[0]) * np.conj(x[1])) / denom
		
		g[0, 0] = c
		g[1, 1] = c
		g[0, 1] = s
		g[1, 0] = np.conj(-s)
		return g

	@staticmethod
	def extract_eigs(M: np.ndarray) -> np.ndarray:
		"""
		Extracts the eigenvalues from the matrix M which is a (quasi) upper triangular matrix received from any functions that produce the Schur decomposition.

		Parameters
		----------
		M :
			A quasi upper triangular square matrix.
		
		Returns
		-------
		``numpy array``:
			The eigenvalues.
		"""
		eigs = list()
		n = M.shape[0]
		i = 0

		while i < n:
			# If the subdiagonal element is close to 
			# 0 or it is the (n - 1, n - 1)th element, 
			# then take the diagonal element.
			if i == n - 1:
				eigs.append(M[i, i])
				break
			# If a diagonal element is chosen, skip 1.
			if abs(np.real(M[i + 1, i])) <= 1e-6:
				eigs.append(M[i, i])
				i += 1
			# If eigenvalues are extracted skip 2 because 2 diagonal elements are added.
			else:
				eigs.extend(ut.eig22(M[i : i + 2, i : i + 2]))
				i += 2

		return np.array(eigs, dtype = np.complex256)

	def qr_hessenberg(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Performs one step of the :math:`QR` algorithm using a hessenberg matrix `H`. Procedurally generates the :math:`QR` decomposition of `H` exploiting the fact that `H` is in hessenberg form.

		Parameters
		----------
		M:	
  			An `n` :math:`\\times` `n` hessenberg 
			matrix.

		Returns
		-------
		:math:`Q`
  		:math:`RQ`
		"""
		givens_matrices = list()
		r = M.copy()
		n = r.shape[0]
		q = np.eye(n, dtype = r.dtype)

		# Get QR decomposition using (g_n g_{n-1} ... g_1 H).
		for i in range(n - 1):
			# Retrieve the Givens Matrix.
			g_ = self.givens_22(r[:, i][i : i + 2])
   
			# Since i and j for the Givens Matrix are consecutive
			# we get a 2 x 2 rotation matrix.
   
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

	def qr_rayleigh_shift(self, eps: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
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
		:math:`U`
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

				# Generate a scaled identity matrix for use in the shifts.
				shift_mat = sigma_k * np.eye(n, dtype = r.dtype)

				# Shift H = H - sigma I.
				r -= shift_mat 
		
				# Perform a step of the QR 
				# hessenberg method.
				q, r = self.qr_hessenberg(r)

				# Shift H = H + sigma I.
				r += shift_mat 
	
				# Form U.
				u = u @ q
				
				k += 1
	
		return u, r
	
	@staticmethod
	def wilkinson_shift(M: np.ndarray) -> Union[complex, float]:
		"""
		A function to compute a stable numerical value of the Wilkison shift (:math:`\\mu`). For a matrix in :math:`\mathbb{C}^{2 \\times 2}`

		.. math::
			\\left[
				\\begin{array}{cc}
					a & b \\\\
					c & d \\\\
				\\end{array}
   			\\right],
      
		.. math::
			\\mu = d - \\frac{b^2\\,\\text{sign}(\\sigma)}{|\\sigma| + \\sqrt{\\sigma^2 + b^2}}
   
		where 
  
		.. math::
			\\sigma = \\frac{1}{2}\\left(a - d\\right).
		
   
		Parameters
		----------
		M:
			A :math:`2 \\times 2` matrix from which the shift is computed.
   
		Returns
		-------
		``Union[complex, float]``
			:math:`\\mu`

		Raises
		------
		ValueError:
			If `M.shape[0] != (2, 2)`. 
		"""
		if M.shape[0] != 2 or M.shape[1] != 2:
			raise ValueError(f"M must be square and of shape (2, 2) but given shape is {M.shape}.")

		sigma = 0.5 * (M[0, 0] - M[1, 1])
		m_2 = M[0, 1] ** 2 
		num = (ut.sign(sigma) * m_2)
		denom = (abs(sigma) + np.sqrt(sigma ** 2 + m_2))
		mu = M[1, 1] - num / denom 
		return mu

	def qr_wilkinson_shift(self, eps: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
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
		:math:`U`
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
				# print(f"Iteration: {i}")
				# if not r[i - 2:, i - 2:].size:
				# 	print(f"Matrix section {i = }")
				# 	with open("qr_output.txt", "a") as f:
				# 		f.write(f"iteration {i}:\n {pd.DataFrame(r).to_string(header = False, index = False)}\n")
    
				# print(f"{r[i - 1 : i + 1, i - 1 : i + 1]}")
				# print(f"{r[i - 1 : i + 1, i - 1 : i + 1].shape}")
     
				# Get shift for each iteration.
				sigma_k = self.wilkinson_shift(r[i - 1 : i + 1, i - 1 : i + 1])
				
				# Generate a scaled identity matrix for use in the shifts.
				shift_mat = sigma_k * np.eye(n, dtype = r.dtype)

				# Shift H.
				r -= shift_mat
		
				# Perform a step of the QR 
				# hessenberg method.
				q, r = self.qr_hessenberg(r)

				# Shift H back.
				r += shift_mat
    
				# Store the transformations.
				u = u @ q
				
				k += 1
	
		return u, r

	def double_shift(self, eps: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Performs an inefficient double shift for a real valued hessenberg matrix H 
		with complex eigenvalues. Calculates the real matrix :math:`M = H^2 - 2\\Re(\\sigma)H + |\\sigma|^2 I` and the double shifted :math:`H_2 = Q^{T} H Q`, where :math:`Q` is from the :math:`QR` decomposition of :math:`M`. Produces the real schur form :math:`H = U^{*}RU`. 
		
		Parameters
		----------
		n_iter:
			Number of double iterations to perform.

		Returns
		-------
		:math:`U`
		:math:`R`

		Raises
		------
		ValueError:
			If `self.H` is not real.
		"""
		if np.iscomplex(self.H).any():
			raise ValueError("Input matrix must be real.")

		H = self.H.copy()
		n = H.shape[0]
		u = np.eye(n, dtype = H.dtype) 
  
		for i in reversed(range(2, n)):
			k = 0
			while abs(H[i, i - 1]) > eps and k < n_iter:
				# Get the shift.
				shift = self.wilkinson_shift(H[i - 1 : i + 1, i - 1 : i + 1])
		
				# Calculate the real matrix M.
				M = H @ H - 2 * shift.real * H + (np.real(shift) ** 2 + np.imag(shift) ** 2) * np.eye(n, dtype = H.dtype)
				
				# QR factorisation of the real matrix M.
				q, r = qr(M)
    
				H = q.T @ H @ q
    
				u = u @ q
    
				k += 1
			
		return u, H
		
	def francis_double_shift(self, eps: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
		"""
		DO NOT USE. DOES NOT FUNCTION CORRECTLY.
		Performs an efficient version of the double shift algorithm to avoid complex
		airthmetic. Produces the Schur decomposition of :math:`H = URU^{*}`. Utilises the *Implicit Q-Theorem* to handle efficient computation of the real matrix :math:`M = H^2 - 2\\Re(\\sigma)H + |\\sigma|^2 I`.
	 
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
		:math:`U`
		:math:`R`
  		"""
		if np.iscomplex(self.H).any():
			raise ValueError("Input matrix must be real.")

		H = self.H.copy()
		n = H.shape[0]
		u = np.eye(n, dtype = H.dtype)
		p = n - 1
  
		while p > 1:
			print(f"Iteration {p = }")
			q = p - 1
   
			s = H[p, p] + H[q, q] 
			t = H[q, q] * H[p, p] - H[q, p] * H[p, q]
   
			# First 3 elements of first column of M.
			x = H[0, 0] ** 2 + H[0, 1] * H[1, 0] - s * H[0, 0] + t
			y = H[1, 0] * (H[0, 0] + H[1, 1] - s)
			z = H[1, 0] * H[2, 1]
   
			for k in range(-1, p - 2):
				print(f"Iteration {k = }")
				t = hg.householder_reflector(np.array([x, y, z], dtype = H.dtype))
				t_norm_squared = t.T @ t
				p_ = np.eye(t.shape[0]) - (2 / t_norm_squared) * (np.outer(t, t.T))
    	
				r = k if k > 1 else 1
				H[k + 1 : k + 4, r :] = p_.T @ H[k + 1 : k + 4, r :]

				r = k + 4 if k + 4 < p else p
				H[: r + 1, k + 1 : k + 4] = H[: r + 1, k + 1 : k + 4] @ p_
    
				# Eigenvector computation.
				u[:, k + 1 : k + 4] = u[:, k + 1 : k + 4] @ p_

				x = H[k + 2, k + 1]
				y = H[k + 3, k + 1]
				z = H[k + 4, k + 1] if k < p - 3 else z

			t = self.givens_22(np.array([x, y], dtype = H.dtype))

			H[q : p + 1, p - 2 :] = t @ H[q : p + 1, p - 2 :]

			H[: p + 1, p - 1 : p + 1] = H[: p + 1, p - 1 : p + 1] @ t.T

			# Eigenvector computation.
			u[:, p - 1 : p + 1] = u[:, p - 1 : p + 1] @ t.T
   
			if abs(H[p, q]) < eps * (abs(H[q, q]) + abs(H[p, p])):
				H[p, q] = 0
				p -= 1
				q = p - 1
	
			elif abs(H[p - 1, q - 1]) < eps * (abs(H[q - 1, q - 1]) + abs(H[q, q])):
				H[p - 1, q - 1] = 0
				p -= 2
				q = p - 1

			print(f"p should have changed {p = }")

		return u, H
