import os


def set_max_num_threads(max_threads):
    """
    Set the maximum number of threads by setting multiple environment variables
    to limit the threads used in parallel/multi-threaded functions. This
    property will limit:

        OPENMP, OPENBLAS, MKL, VECLIB and NUMEXPR

    If max_num_threads is set to a value < 0, environment variables are not
    updated.

    This function must be called before the dipole_inverse module to take
    effect
    """
    max_num_threads = str(max_threads)
    if int(max_threads) > 0:
        os.environ["OMP_NUM_THREADS"] = max_num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = max_num_threads
        os.environ["MKL_NUM_THREADS"] = max_num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = max_num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = max_num_threads
