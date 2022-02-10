"""
Utility
=======
"""

from typing import Union, Tuple
from scipy.linalg import eig
import numpy as np
import pandas as pd

def complex_matrix(n: int, a: float, b: float, type_: np.dtype = np.complex256) -> np.ndarray:
	"""
	Produces a random `n` :math:`\\times` `n` complex square matrix. The absolute values of all values in the matrix range between `2a` and `2b`.

	Parameters
	----------
	n:	
		Square matrix size.
	a:
		Lower limit for random number generator.
	b: 
		Upper limit for random number generator.
	type_:
		Complex or real numpy `dtype`.
  
	Returns
	-------
	`numpy ndarray`:
		A complex square matrix. 
  
	Raises
	------
	ValueError
		If `b` :math:`\\leq` `a`.
	"""
	
	if a >= b:
		raise ValueError("Required: b > a")
	
	r = (b - a) * np.random.default_rng().random(size = (n, n)) + a
	c = (b - a) * np.random.default_rng().random(size = (n, n)) + a
	m = r + 1j * c
	
	return m.astype(type_)

def sign(z: complex) -> Union[complex, float]:
	"""
	A general sign function for complex valued inputs.
	
	Parameters
	----------
	z:
		A complex number.
  
	Returns
	-------
	complex:
		1 if `z` = 0 otherwise `z / |z|`.
	"""
	if z == 0:
		return 1
	return z / abs(z)


def eig22(M: np.ndarray) -> Tuple[Union[complex, float], Union[complex, float]]:
	"""
	Computes the eigenvalues of the :math:`2 \\times 2` matrix `M`.
	
	Parameters
	----------
	M:
		A :math:`2 \\times 2` complex matrix.
  
	Returns
	-------
	`eig_1`
	`eig_2`
 
	Raises
	------
	ValueError:
		If `M` is not square and of shape :math:`2 \\times 2`.
	"""
	if M.shape[0] != 2 or M.shape[1] != 2:
		raise ValueError(f"Input matrix must be of shape (2, 2) but is of shape {M.shape}.")

	m = 0.5 * (M[0, 0] + M[1, 1])
	d = M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1] 
  
	# Special square root function that returns complex values when the input is negative.
	sqrt_disc = np.emath.sqrt(m ** 2 - d)
 
	eig_1 = m + sqrt_disc
	eig_2 = m - sqrt_disc
 
	return m + sqrt_disc, m - sqrt_disc
 
	


def closeness(actual: np.array, desired: np.array, tol: float) -> Tuple[bool, Union[pd.DataFrame, None]]:
	"""
	Judges if arrays are close to each other upto a certain `tol` using the equation
	
	.. math::

		|a - b| \\leq tol
	Parameters
	----------
	actual:
		The array with predicitons.
	desired:
		The array with desired values.
  
	Returns
	-------
	A tuple consisting of a boolean indicating if all values are close to each other and a dataframe containing the mismatched elements, if any.
 	"""
	mismatched_positions = list()
	for predicted, val in zip(actual, desired):
		if abs(predicted - val) > tol:
			mismatched_positions.append(False)
		mismatched_positions.append(True)
			
	mismatched_elements = None
 
	if not all(mismatched_positions):
		mismatched_elements = pd.DataFrame(np.vstack([desired[mismatched_elements], actual[mismatched_elements]]).T, columns = ["Real Values", "Predicted Values"])
			
	return all(mismatched_positions), mismatched_elements


def main():
	m = np.array([[1 + 2j, 1], [-2, 3]], dtype = np.complex64)
	print(eig22(m))
	print(eig(m)[0])
	
	
if __name__ == "__main__":
	main()