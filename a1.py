#!/usr/bin/env python3

""" Data Structures and Algorithms for CL III, Assignment 1
    See <https://dsacl3-2019.github.io/a1/> for detailed instructions.

    <Please insert your name and the honor code here.>
"""

import numpy as np
import time
from scipy.stats import norm, multivariate_normal
from numpy.random import uniform

def find_mode_linear(seq):
    """ Find a mode in a sample with unique items (numbers).

    Finding any mode (not necessarily with the maximum value) is
    acceptable. However, make sure that your implementation does not
    do unnecessary work.

    Parameters:
    seq: A sequence with unique items with well-defined < and > relations.

    Returns:
    The index of the sequence where a mode is found.

    """
    if len(seq) == 0:
        return None
    if len(seq) == 1 or seq[0] > seq[1]:
        return 0
    for i in range(1,len(seq)-1):
        if seq[i - 1] < seq[i] and seq[i] > seq[i + 1]:
            return i
    return len(seq)-1

def find_unique_mode(seq):
    """ Find the unique mode in a sample with unique items (numbers).

    Make sure that your this implementation (on average, asymptotically)
    faster than find_mode_linear().

    Parameters:
    seq: A sequence with unique items with well-defined < and > relations.
         The sequence is required to have a unique mode.

    Returns:
    The index of the sequence where a mode is found.

    """
    if len(seq) == 0:
        return None
    if len(seq) == 1 or seq[0] > seq[1]:
        return 0
    left, right = 0, len(seq) - 1
    while left < right:
        mid = (left + right) // 2
        if seq[mid] < seq[mid + 1]:
            left = mid + 1
        elif seq[mid] > seq[mid + 1]:
            right = mid
    return left

def time_search(search_func, n_samples=1000, dim=1000):
    """ Return the average running time of the given function on random data.

    This function runs the search_func n_samples times with
    random samples of size 'dim' as parameter. The random samples
    should have only a single mode/peak. You are free to choose the
    range and the way you construct the array. Make sure you are
    timing only the search function, not the preperation of the random
    array.

    Parameters:
    search_func     The function to run
    n_samples       Number of times the function is run
    dim             Dimensions of the random array

    Return value: The average running time of the function over
                  multiple runs.
    """
    t = 0.0
    for _ in range(n_samples):
        loc = uniform(-2,2)
        # nothing special about the normal distribution, it is just a
        # unimodal distribution. Forming a single-peek sequence of any
        # sort (e.g., creating two sorted arrays anc concetaneting
        # them) should be fine
        sample = norm.pdf(sorted(uniform(-1,1, dim)), loc)
        start = time.time()
        search_func(sample)
        t += time.time() - start
    return t / n_samples


def find_mode2d(data):
    """Find a mode of 2D data sample.

    Parameters:
    data     The 2D array (matrix) of numbers (or any object with
             < and > relations). You are strongly recommended to 
             use numpy arrays, as input, but a list-of-lists structure
             is also fine.

    Return value: the index of the mode - a tuple (row,col)
    """

    data = np.array(data) # in case another 2D structure is given

    # sanity checks (not really needed, but a good idea)
    assert len(data.shape) == 2, "A 2D array is required."
    if data.size == 0:
        return None

    i, j = 0, 0 # arbitrary any valid index should work
    # unconditional loop are scary - if input is not as expected,
    # it may loop forever. It is a good idea to be more robust (left
    # as an exercise ;).
    while True:
        left, right = max(j - 1, 0), min(j + 1, data.shape[1] - 1)
        top, bottom = max(i - 1, 0), min(i + 1, data.shape[0] - 1)

        # Find the maximum value in the neighborhood of [i,j]
        # Note that with a bit of work, we can avoid comparisons to
        # the previous window. It could also be neater with np.argmax.
        # The explicit loops are here for simplicity of demonstration.
        maxval, max_i = data[top, left], (top, left)
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                if data[row, col] > maxval:
                    maxval = data[row, col] 
                    max_i = row, col
        if (i, j) == max_i:
            return i, j
        i, j = max_i

if __name__ == "__main__":


#     for dim in 5,100,1000,10000:
#         print(dim, "dimensions")
#         print('\tlinear search: {:.10f}'.format( 
#                 time_search(find_mode_linear,dim=dim)))
#         print('\tbinary search: {:.10f}'.format(
#                 time_search(find_unique_mode,dim=dim)))
# 
#     for _ in range(1000):
#         x, y = np.mgrid[-2:2:00.1,-2:2:00.1]
#         xshift, yshift = uniform(-2.5,2.5), uniform(-2.5,2.5)
#         dist = multivariate_normal.pdf(
#                             np.dstack((x+xshift,y+yshift)), mean = [0,0])
#         maxx = find_mode2d(dist)
#         maxnp = np.unravel_index(dist.argmax(), dist.shape)
#         assert maxx == maxnp
