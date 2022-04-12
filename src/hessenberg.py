"""
Hessenberg
==========
"""
import numpy as np 
import pandas as pd 
from scipy.linalg import hessenberg, norm

import utility as ut

from typing import Tuple, Union 


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
    ``numpy array``:
        The Householder vector. 
    """
    u = x.copy()

    if any(np.iscomplex(u)):
        rho = -np.exp(1j * np.angle(u[0]), dtype = u.dtype)
    else:
        rho = -ut.sign(u[0])

    # Set the Householder vector
    # to u = u \pm alpha e_1 to 
    # avoid cancellation.
    u[0] -= rho * norm(u)
  
    # Vector needs to have 1 
    # in the 2nd dimension.
    return u.reshape(-1, 1)
    
def hessenberg_transform(M: np.ndarray, calc_u: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], None]:
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
    if ut.hessenberg_q(M):
        print("Input matrix is already Hessenberg.")
        return None

    h = M.copy()
    n = h.shape[0]
    householder_vectors = list()
    skipped = set()
   
    for l in range(n - 2):

        # Get the Householder vector.
        t = householder_reflector(h[l + 1 :, l])

        # Norm**2 of the Householder vector.
        t_norm_squared = np.conj(t).T @ t
        
        if t_norm_squared == 0.0:
            factor = 0.0
        else:
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