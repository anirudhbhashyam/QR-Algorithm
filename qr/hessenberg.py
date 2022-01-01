import numpy as np 
import pandas as pd 
from scipy.linalg import hessenberg
  
def balance(M: np.ndarray):
	# Implement a balancing algorithm to
	# take care of rounding errors.
	pass

def householder_reflector(i: int, x: np.array):
	"""
	Produces the Householder
	vector based on the input 
	vector x and position i. 
	The householder vector acts
	to reduce the ith component
	and all subsequent components 
	of x to 0.
 
	|a_1|		|alpha|	
	|a_2|	->	|0|
	|a_3|		|0|

	Parameters
	----------
	x:	A numpy array who's entries
		after the ith index need to 
		be 0ed. 
	i: 	An int specifying the index.

	Returns
	-------
	A numpy array that acts as the 
	Householder vector. 
	"""
	u = x.copy()
	n = u.shape[0]

	# Norm of the input vector.
	delta = np.linalg.norm(u)
	rho = np.exp(-1j * np.angle(u[0]))

	# Set the Householder vector
	# to u = u \pm alpha e_1 to 
	# avoid cancellation.
	e_1 = np.zeros(n); e_1[0] = 1
	u[0] -= rho * delta
	
	# Scaling factor.
	factor = np.linalg.norm(u - rho * delta * e_1) 
 
	return (1 / factor) * u
	
def hessenberg_transform(M: np.ndarray) -> np.ndarray:
	"""
	Converts a given matrix to 
	Hessenberg form using
	Houeholder transformations.

	Parameters
	----------
	M:	A complex square 
		numpy 2darray.

	Returns
	-------
	A tuple consisting of numpy
 	2-D arrays which are the 
	hessenberg form and the 
	permutation matrix.
	"""
	h = M.copy()
	householder_vectors = list()
	n = h.shape[0]
	u = np.eye(n)
 
	for i in range(0, n - 2):
		# Get the Householder vector.
		t = householder_reflector(i, h[i + 1:, i])
		# print(f"h = {pd.DataFrame(h)}")
  
		# Norm of the Householder vector.
		t_norm_squared = t.conj().T @ t
  
		# Form the Householder matrix.
		p = np.eye(h[i + 1:, i].shape[0]) - 2 * (np.outer(t, t)) / t_norm_squared
		
		# Resize and refactor the Householder matrix.
		p = np.pad(p, ((i + 1, 0), (i + 1, 0)), mode = "constant", constant_values = ((0, 0), (0, 0)))
		for k in range(i + 1):
			p[k, k] = 1

		# Perform a similarity transformation on h
		# using the Householder matrix.
		h = p @ h @ p

		householder_vectors.append(t)
		
	# Store the linear transformations.
	for i in reversed(range(0, n - 2)):
		u = p @ u

	return h, u

if __name__ == "__main__":
	M = np.array([[14, 15, 10, 18, 19, 18, 15, 15], [12, 10, 17, 11, 20, 20, 15, 
  12], [11, 19, 19, 16, 17, 18, 17, 12], [11, 12, 18, 18, 18, 19, 14, 
  20], [16, 12, 16, 10, 19, 17, 12, 16], [17, 13, 10, 18, 14, 14, 15, 
  17], [11, 11, 14, 20, 20, 19, 20, 13], [16, 11, 14, 12, 16, 13, 17, 
  17]], dtype = np.float32)
	print(f"Original matrix:\n {pd.DataFrame(M)}")
	print(f"Hessenberged:\n {pd.DataFrame(hessenberg_transform(M)[0].round(4))}")
	w, v = np.linalg.eig(hessenberg_transform(M)[0])
	print(f"Eigenvalues original: {w}") 
 
