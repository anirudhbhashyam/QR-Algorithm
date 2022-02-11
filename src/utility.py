"""
Utility
=======
"""
from typing import Union, Tuple, Iterable, Any

import numpy as np
import pandas as pd
from scipy.linalg import eig


def complex_matrix(n: int, a: float, b: float, type_: np.dtype = np.complex256) -> np.ndarray:
	"""
	Produces a random `n` :math:`\\times` `n` complex square matrix. The absolute real and complex parts of allof all values in the matrix range between `2a` and `2b`.

	Parameters
	----------
	n:	
		Square matrix size.
	a:
		Lower limit for random number generator.
	b: 
		Upper limit for random number generator.
	type\\_:
		Complex numpy type.
  
	Returns
	-------
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

def sign(z: Union[complex, float]) -> Union[float, complex]:
	"""
	A general sign function for complex valued inputs.
	
	Parameters
	----------
	z:
		A complex number.
  
	Returns
	-------
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
	First eigenvalue.
	Second eigenvalue.

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

def closeness(actual: Iterable, desired: Iterable, tol: float) -> Tuple[bool, Union[pd.DataFrame, None]]:
	"""
	Judges if arrays are close to each other upto a certain `tol` using the equation. For arrays :math:`x` and :math:`y`, the function checks
	
	:math:`|a - b| \\leq tol`
	for each positional pair :math:`a \\in x` and :math:`b \in y`.
 
	Parameters
	----------
	actual:
		The array with predicitons.
	desired:
		The array with desired values.
	tol:
		The error tolerance.
  
	Returns
	-------
	Boolean value indicating if all elements of both arrays are close.
	Mismatched elements, if any.
 
	Raises
	------
	ValueError:
		If `actual` and `desired` array lengths are not equal.
 	"""
	if len(actual) != len(desired):
		raise ValueError(f"Length of input arrays do not match, actual is {len(actual)} and desired is {len(desired)}.")

	b = True
	mismatched_positions = list()
	for i, (predicted, val) in enumerate(zip(actual, desired)):
		if abs(predicted - val) > tol:
			b = False
			mismatched_positions.append(i)
			
	mismatched_elements = None
	
	actual = np.array(actual)
	desired = np.array(desired)
 
	if not b:
		mismatched_elements = pd.DataFrame(np.vstack([desired[mismatched_positions], 
												actual[mismatched_positions], 
												desired[mismatched_positions] - actual[mismatched_positions]]).T,columns = ["Real Values", "Predicted Values", "Difference"])
			
	return b, mismatched_elements


def main():
	a = [1.00, 2.00, 3.00]
	b = [1.01, 2.02, 3.00]
	print(closeness(a, b, 1e-2)[0])
	
	
if __name__ == "__main__":
	main()