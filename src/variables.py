import os


# Path specification for test matrices 
# from matrix market. scipy.io.mmread
# is used to read gunzipped matrix files.
MATRIX_MARKET_PATH = os.path.abspath("../test_matrices")
MATRIX_MARKET_FILE_EXT = "mtx.gz"