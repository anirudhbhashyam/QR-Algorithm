import numpy as np 
import pandas as pd 
from scipy.linalg import hessenberg

BLOCK_SIZE = 8
  
def balance(M: np.ndarray):
	# Implement a balancing algorithm to
	# take care of rounding errors.
	pass

def householder_reflector(x: np.array):
	"""
	Produces the Householder
	vector based on the input 
	vector x. The householder 
 	vector acts as:
 
	|a_1|		|alpha|	
	|a_2|	->	|0|
	|a_3|		|0|

	Parameters
	----------
	x:	
		A numpy array who's entries
		after the ith index need to 
		be 0ed. 
	i: 	
 		An int specifying the index.

	Returns
	-------
	A numpy array that acts as the 
	Householder vector. 
	"""
	u = x.copy()
	n = u.shape[0]

	# Norm of the input vector.
	delta = np.linalg.norm(u)
	rho = -np.exp(1j * np.angle(u[0]))

	# Set the Householder vector
	# to u = u \pm alpha e_1 to 
	# avoid cancellation.
	e_1 = np.zeros(n)
	e_1[0] = 1.0
	 
	u[0] -= rho * delta
 
	# Vector needs to have 1 
	# in the 2nd dimension.
	return u.reshape(-1, 1)
	
def hessenberg_transform(M: np.ndarray) -> np.ndarray:
	"""
	Converts a given matrix to 
	Hessenberg form using
	Houeholder transformations.

	Parameters
	----------
	M:	
 		A complex square 
		numpy 2darray.

	Returns
	-------
	A tuple consisting of numpy
 	2-D arrays which are the 
	hessenberg form and the 
	permutation matrix.
	"""
	h = M.copy()
	
	n = h.shape[0]
	u = np.eye(n, dtype = np.complex128)
	householder_vectors = list()
 
	TILE_SHAPE = 2
	BLOCK_SIZE = h.shape[0] // TILE_SHAPE
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
		t_norm_squared = np.dot(t.conj().T, t)

		# Perform a similarity transformation on h
		# using the Householder matrix.
		# h = p @ h @ p.

		# Left multiplication by I - 2uu^{*}
		h[l + 1 :, l :] -= 2 * t * (np.dot(t.conj().T, h[l + 1 :, l :]) / t_norm_squared)
		# Right multiplication by I - 2uu^{*}
		h[ :, l + 1 :] -= 2 * t.conj().T * (np.dot(h[ :, l + 1 :], t) / t_norm_squared) 

		householder_vectors.append(t)
			
	# Store the transformations.
	for k in reversed(range(n - 2)):
		t = householder_vectors[k]
		t_norm_squared = np.dot(t.conj().T, t)
		u[k + 1 :, k + 1 :] = 2 * t * (np.dot(t.conj().T, u[k + 1 :, k + 1 :]) / t_norm_squared)

	return h, u

if __name__ == "__main__":
	M = np.array([[14, 15, 10, 18, 19, 18, 15, 15], 
               [12, 10, 17, 11, 20, 20, 15, 12], 
               [11, 19, 19, 16, 17, 18, 17, 12], 
               [11, 12, 18, 18, 18, 19, 14, 20], 
               [16, 12, 16, 10, 19, 17, 12, 16], 
               [17, 13, 10, 18, 14, 14, 15, 17], 
               [11, 11, 14, 20, 20, 19, 20, 13], 
               [16, 11, 14, 12, 16, 13, 17, 17]], dtype = np.complex128)
	print(f"Original matrix:\n {pd.DataFrame(M)}")
	print(f"Hessenberged:\n {pd.DataFrame(hessenberg_transform(M)[0].round(4))}")
	print(f"Hessenberged (scipy):\n {pd.DataFrame(hessenberg(M).round(4))}")
	# print(f"Test equality: {np.allclose(hessenberg_transform(M)[0], hessenberg(M), rtol = 1e-6)}")
	# w, v = np.linalg.eig(hessenberg_transform(M)[0])
	# print(f"Eigenvalues original: {w}") 
 
