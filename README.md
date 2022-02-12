# QR-Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Prerequisites
Running the prediction model requires these basic dependancies:

* Python `>=3.8.0`
* Numpy `>= 1.21.0`
* Pandas `>= 1.3.0`
* Sphinx (for building documentation) `>=4.3.0`.

Detailed environment information can be found in `requirements.txt`.

# Set up
* `git clone --recursive https://github.com/anirudhbhashyam/QR-Algorithm`
* `cd QR-Algorithm`

# Use
First and foremost it is recommended to run `usage.ipynb` and see the functionality in action.

Since this repository is not a package, the source directory will have to be included in the environment variable PYTHONPATH when **direct import statements** want to be used. 

In a python script, the following must be added:
```
import sys
import os
sys.path.append(os.path.abspath(<path_to_src>))
```

To use a `module` present in `src` in a python script, either the above method can be used and then one can use `import hessenberg` and such forth, or indirect import statements can be used such as `from <path_to_src> import <module> `.

## Hessenberg
The `hessenberg` module provides two functions:
* `householder_reflector`: Produces Householder vectors.
* `hessenberg_transform`: Produces the Hessenberg form.
Implementation details can be found in the documentation. 

## QR
The `qr` module provides an interface `QR` which accepts a complex/real matrix when it is initialised and stores a the matrix in hessenberg form. The main functions in this class are 
* `qr_hessenberg`: Generates the $QR$ decomposition of a hessenberg matrix. 
* `rayleigh_shift`: Performs the Rayleigh shift iteration on a matrix in hessenberg form.
* `wilkinson_shift`: Performs the Wilkinson shift iteration on a matrix in hessenberg form.
* `double_shift`: Perform an inefficient double shift on a real matrix in hessenberg form.

Implementation details can be found in the documentation. 

# Testing
To run all tests one can do the following from the top level directory:
`python -m unittest discover tests`
or
* `cd tests`
* `python -m unittest test_*.py`

# Documentation 
https://qr-algorithm.readthedocs.io/en/latest/