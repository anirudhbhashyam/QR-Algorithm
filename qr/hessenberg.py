"""
Hessenberg
==========
"""
import numpy as np 
import pandas as pd 
from scipy.linalg import hessenberg, norm

import utility as ut

from typing import Tuple 


def balance(M: np.ndarray):
	# Implement a balancing algorithm to
	# take care of rounding errors.
	pass

def householder_reflector(x: np.array) -> np.array:
	"""
	Produces the Householder
	vector based on the input 
	vector `x`. The householder 
 	vector :math:`u` acts as
  
  	.. math:: 
   		(I-2uu^{*})x = 
		\\left[
			\\begin{array}{c}
				\\alpha \\\\
				0 \\\\
				\\vdots \\\\
				0 \\\\
			\\end{array}
		\\right]

	Parameters
	----------
	x:	
		A complex `numpy array` who's entries
		after the 1st element need to 
		be 0ed. 
  
	Returns
	-------
	`numpy array`:
		The Householder vector. 
	"""
	u = x.copy()

	if any(np.iscomplex(u)):
		rho = -np.exp(1j * np.angle(u[0]), dtype = np.complex256)
	else:
		rho = -ut.sign(u[0])

	# Set the Householder vector
	# to u = u \pm alpha e_1 to 
	# avoid cancellation.
	u[0] -= rho * norm(u)
 
	# Vector needs to have 1 
	# in the 2nd dimension.
	return u.reshape(-1, 1)
	
def hessenberg_transform(M: np.ndarray, calc_u: bool = True) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Converts a given complex square matrix to Hessenberg form using Householder transformations.

	Parameters
	----------
	M:	
 		A complex square `numpy ndarray`.
	calc_u:
		Flag to determine whether to calculate the transfomation matrix.

	Returns
	-------
	`numpy ndarray`:
		The transformed matrix
	`numpy ndarray`:
		The permutation matrix if `calc_q = True`.
	"""
	h = M.copy()
	n = h.shape[0]
	householder_vectors = list()
 
	# TILE_SHAPE = 2
	# BLOCK_SIZE = h.shape[0] // TILE_SHAPE
	# for i in range(0, n, BLOCK_SIZE):
	# 	for j in range(0, n, BLOCK_SIZE):
	# 		# print(f"({i}:{i + BLOCK_SIZE}, {j}:{j + BLOCK_SIZE})")
	# 		h_blocked = h[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
	# 		# print(f"{h_blocked = }")
	# 		u_blocked = u[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
   
	# 		n_ = h_blocked.shape[0]
	# 		householder_vectors = list()
   
	for l in range(n - 2):
		# Get the Householder vector.
		t = householder_reflector(h[l + 1 :, l])

		# Norm**2 of the Householder vector.
		t_norm_squared = t.conj().T @ t
  
		# p = np.eye(h[l + 1:, l].shape[0]) - 2 * (np.outer(t, t)) / t_norm_squared

		# # # Resize and refactor the Householder matrix.
		# p = np.pad(p, ((l + 1, 0), (l + 1, 0)), mode = "constant", constant_values = ((0, 0), (0, 0)))
		# for k in range(l + 1):
		# 	p[k, k] = 1

		# # Perform a similarity transformation on h
		# # using the Householder matrix.
		# h = p @ h @ p
  
		factor = 2.0 / t_norm_squared
  
		# Left multiplication by I - 2uu^{*}.
		h[l + 1 :, l :] -= factor * (t @ (t.conj().T @ h[l + 1 :, l :]))
  
		# Right multiplication by I - 2uu^{*}.
		h[ :, l + 1 :] -= factor * ((h[ :, l + 1 :] @ t) @ t.conj().T)
  
		# Force elements below main
		# subdiagonal to be 0.
		h[l + 2 :, l] = 0.0

		# Store the transformations 
		# to compute u.
		householder_vectors.append(t)
  
	# Calculate transfomation matrix
	# from the stored transformations.
	if calc_u:
		u = np.eye(n, dtype = M.dtype)
		for k in reversed(range(n - 2)):
			t = householder_vectors[k]
			t_norm_squared = np.dot(t.conj().T, t)
			u[k + 1 :, k + 1 :] = 2 * t * (t.conj().T @ u[k + 1 :, k + 1 :]) / t_norm_squared
		return h, u

	return h

if __name__ == "__main__":
	n = 10
	a = 10.0
	b = 20.0
	M = ut.complex_matrix(n, a, b)
	# M = np.array([[14, 15 + 2j, 10, 18, 19, 18, 15, 15], 
	#            [12, 10, 17, 11, 20, 20, 15, -12], 
	#            [11, 19, 19, -16, 17, 18, 17, 12], 
	#            [11, 12, 18, 18, 18, 19, 14, 20], 
	#            [16, 12, 16, 10, 19, 17, 12, 16], 
	#            [17, 13, -10, 18, 14, 14, 15, 17], 
	#            [11, 11, 14, 20, 20, 19, 20, 13], 
	# 			[16, 11, 14, 12, 16, 13, 17, 17]], dtype = np.complex128)
	# max_el = np.max(M)
	# M /= max_el
	# M = M.astype(np.float128)
	# print(M.dtype)
	print(f"Original matrix:\n {pd.DataFrame(M)}")
	hess_from_alg, _ = hessenberg_transform(M)
	hess_from_scipy = hessenberg(M) 
	print(f"Hessenberged:\n {pd.DataFrame(hess_from_alg)}")
	print(f"Hessenberged (scipy):\n {pd.DataFrame(hess_from_scipy)}")
	print(f"Test equality: {np.allclose(hess_from_alg, hess_from_scipy, rtol = 1e-6)}")
	# w, v = np.linalg.eig(hessenberg_transform(M)[0])
	# print(f"Eigenvalues original: {w}") 
 
