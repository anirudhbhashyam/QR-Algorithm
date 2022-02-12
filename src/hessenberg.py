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
	Converts a given complex square matrix to Hessenberg form using Householder transformations. Produces the matrices H and U such that :math:`M = UHU^{*}`, where :math:`H` is in hessenberg form and :math:`U` is a unitary matrix.

	Parameters
	----------
	M:	
 		A complex square matrix.
	calc_u:
		Flag to determine whether to calculate the transfomation matrix.

	Returns
	-------
	H
	U
	"""
	h = M.copy()
	n = h.shape[0]
	householder_vectors = list()
   
	for l in range(n - 2):
		# Get the Householder vector.
		t = householder_reflector(h[l + 1 :, l])

		# Norm**2 of the Householder vector.
		t_norm_squared = np.conj(t).T @ t
  
		factor = 2.0 / t_norm_squared
  
		# Left multiplication by I - 2uu^{*}.
		h[l + 1 :, l :] -= factor * (t @ (np.conj(t).T @ h[l + 1 :, l :]))
  
		# Right multiplication by I - 2uu^{*}.
		h[:, l + 1 :] -= factor * ((h[:, l + 1 :] @ t) @ np.conj(t).T)
  
		# Force elements below main
		# subdiagonal to be 0.
		h[l + 2 :, l] = 0.0

		# Store the transformations 
		# to compute u.
		householder_vectors.append((factor, t))
  
	# Calculate transfomation matrix
	# from the stored transformations.
	if calc_u:
		u = np.eye(n, dtype = M.dtype)
		for i in reversed(range(n - 2)):
			factor, t = householder_vectors[i]
			# p = np.eye(n, dtype = M.dtype) - 2 * np.outer(t, t)
			u[i + 1 :, i + 1 :] -= factor * (t @ (np.conj(t).T @ u[i + 1 :, i + 1 :]))
			# u = p @ u
		return h, u

	return h


def main():
    # -- DO NOT UNCOMMENT -- #
	# n = 5
	# a = 10.0
	# b = 20.0
	# m = ut.complex_matrix(n, a, b)
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
	# print(f"Original matrix:\n {pd.DataFrame(m)}")
	# h, u = hessenberg_transform(m) 
	# print(f"Hessenberg transformed:\n {pd.DataFrame(h)}")
	# print(f"Transformation matrix:\n {pd.DataFrame(u)}")
	# print(pd.DataFrame(u @ h @ u.conj().T - m))
	# w, v = np.linalg.eig(hessenberg_transform(M)[0])
	# print(f"Eigenvalues original: {w}") 
	pass
    

if __name__ == "__main__":
	main()