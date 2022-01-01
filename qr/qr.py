import numpy as np
import pandas as pd
from scipy.linalg import companion, qr, hessenberg

from hessenberg import hessenberg_transform


class QR:
	"""
	A simple implementation of the QR algorithm
	and its variations.
	"""
	
	def __init__(self, M: np.ndarray):
		if M.shape[0] != M.shape[1]:
			raise AttributeError("Matrix must be square.")

		# Store matrix in Hessenberg form.
		self.H = hessenberg(M)

	@staticmethod
	def givens_rotation(i: int, j: int, x: np.array, n: int) -> np.ndarray:
		"""
		Generates an (n x n) special 
		Givens Rotation matrix based on 
		the parameters i, j, x, and n. 
		The rotation matrix acts to reduce 
  		the jth component of the vector 
		x to 0.

		Parameters
		----------
		i:	ith row.
		j:	jth column.
		x: 	Vector who's jth
			entry needs to be 
			reduced to 0.
		n: 	Size of the returned
			Givens Matrix.

		Returns
		-------
		A Givens Rotation matrix as a numpy 
		ndarray.
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

	def qr_hessenberg(self, M: np.ndarray) -> (np.ndarray, np.ndarray):
		"""
		Performs one step of the 
		QR algorithm using a hessenberg
		matrix H. Procedurally generates 
  		the QR decomposition of H
		exploiting the fact that H is 
		hessenberg.

		Parameters
		----------
		self:	An n x n hessenberg 
			matrix.

		Returns
		-------
		Two numpy ndarrays
		RQ and Q, where H = QR
		is the QR decomposition of
		H.
		"""
		# Improvements: 
		# Generate the Givens Matrix once.

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
			
			# Apply the Givens Matrix
			# (conjugated and transposed)
			# from the left.
			r = g.conj().T @ r

		# Get new H using (R g_1 g_2 ... g_n).
		for i in range(n - 1):
			# Apply the Givens Matrix to r 
			# from the right.
			r = r @ givens_matrices[i] 

		return r, q

	def qr_rayleigh_shift(self, eps: float, n_iter: int) -> (np.ndarray, np.ndarray):
		"""
		Performs the QR algorithm 
		employing the hessenberg method
		for the decomposition and utilises 
		the Rayleigh shift. 
		
		Parameters
		----------
		eps:    Tolerance to break 
				matrix self.H.
		n_iter: Number of iterations to
				perform.
		
		Returns
		-------
		Two numpy ndarrays U, R 
		where self.H = U R U^* is the Schur 
		decomposition of self.H.
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

				# Update H and U.
				r += sigma_k * np.eye(n)
				u = u @ q
				
				# Update iteration counter.
				k += 1
	
		return (u, r)

	def double_shift(self, shift: complex):
		"""
		Performs an inefficient
		double shift for a real valued 
		hessenberg matrix H 
		with complex eigenvalues.
		
		Parameters
		----------
		H:	
			A real numpy ndarray.
		shift:				
			Approximated complex eigenvalue.

		Returns
		-------
		A numpy ndarray, H_2
		of the algorithm.
		"""
		H = self.H.copy()
  
		# Calculate the real matrix M.
		M = H @ H - 2 * shift.real * H + (np.abs(shift) ** 2) * np.eye(H.shape[0])
		
		# QR factorisation of M.
		# Perform a step of the QR  
		# hessenberg method.
		r, q = self.qr_hessenberg(M)
		
		return q
		
if __name__ == "__main__":
	M = np.array([
					[4, 2, 6, 3, 4, 5, 1, 2],
					[7, 3, 8, 2, 8, 2, 6, 5],
					[8, 9, 6, 8, 7, 9, 8, 8],
					[3, 2, 2, 4, 9, 1, 1, 8],
					[6, 8, 1, 6, 4, 1, 2, 2],
					[8, 2, 1, 5, 2, 5, 5, 1],
					[3, 5, 7, 9, 8, 2, 10, 5],
					[8, 6, 10, 7, 8, 8, 4, 7]
				], dtype = complex)
	
	pd.options.display.max_columns = 200
	pd.set_option("display.width", 1000)
 
	print(f"Original matrix:\n {pd.DataFrame(M)}")
	qr_alg = QR(M)
	print(f"Hessenberged:\n {pd.DataFrame(qr_alg.H)}")
	r = qr_alg.double_shift(4.0 + 4.j)
	
	print(f"U:\n {pd.DataFrame(np.diag(r))}")
	# print(f"R:\n {pd.DataFrame(q)}")
	# u, r = qr_alg.qr_rayleigh_shift(1e-12, 20)
	# pd.set_option('display.max_columns', None)
	# print(pd.DataFrame(u))
	# print(pd.DataFrame(r))