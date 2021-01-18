#!/usr/bin/env python3

""" Data Structures and Algorithms for CL III, Assignment 3
    See <https://dsacl3-2019.github.io/a2/> for detailed instructions.

    <Please insert your name and the honor code here.>
"""

import random
import time

def insertion_sort(seq, L=None, R=None):
    if L is None:
        L, R = 0, len(seq) - 1
    for i in range(L, R +1):
        cur = seq[i]
        j = i
        while seq[j - 1] > cur and j in range(1,i+1):
            seq[j] = seq[j - 1]
            j -= 1
        seq[j] = cur

def qsort(seq, L=None, R=None, cutoff=None):
    """Sort the list from S[L] to S[R] inclusive using the quicksort algorithm
        in-place with a median-of-three approach.
        
    Parameters
    ----------
    seq:      The sequence to be sorted.
    L:      The leftmost index to consider. If None, should be set to 0.
    R:      The rightmost index to consider. If None, should be set to
            len(seq) - 1.
    cutoff: If given, and (sub)sequence to be sorted is less than 'cutoff',
            use insertion sort (note: it is unlikely to get any
            improvement in this setting, but this is a common trick
            some other algorithms).
    Returns None, modifies the sequence given as input (in place).
    """ 
    
    if L is None:
        L, R = 0, len(seq) - 1
    if R - L  < 1: return
    if cutoff and R - L < cutoff:
        insertion_sort(seq)
        return 

    M = (L+R)//2 # get the midpoint of the list
    # We want to move the middle element to the end, this way,
    # the algorithm will be identical to the one described
    # in the book.
    if seq[L] < seq[M] < seq[R] or seq[R] < seq[M] < seq[L]:
        seq[R], seq[M] = seq[M], seq[R]
    elif seq[M] < seq[L] < seq[R] or seq[R] < seq[L] < seq[M]:
        seq[R], seq[L] = seq[L], seq[R]

    left = L 
    right = R - 1
    
    while left <= right: 
        while left <= right and seq[left] < seq[R]: 
            left += 1 
        while left <= right and seq[R] < seq[right]:
            right -= 1
            
        if left <= right:
            seq[left], seq[right] = seq[right], seq[left]
            left += 1 
            right -= 1   
    seq[left], seq[R] = seq[R], seq[left] 
    
    qsort(seq, L, left - 1)
    qsort(seq, left + 1, R)


def bucket_sort(words, index=0, chmin=None, chmax=None):
    """Sort given set of strings using bucket sort.

    This is not part of the required implementation/interface, but it
    is a reasonable way to split the labor between radix sort and
    bucket sort.

    Parameters
    ----------
    words:  The sequence of words to be sorted
    index:  The index of the character that sorting will be based on.
            ["abc", "bac", "cba"] is sorted as ["bac", "abc", "cba"]
            if index = 1, and as ["cba", "bac", "abc"] if index=2.
    chmin:  The minumum character to consider
    """
    if chmin is None: chmin = 'a'
    if chmax is None: chmax = 'z'
    buckets = [[] for _ in range(ord(chmin), ord(chmax) + 2)]
    for word in words:
        if index >= len(word): # word is shorter
            ch_i = 0
        elif ord(chmin) <= ord(word[index]) <= ord(chmax):
            ch_i = ord(word[index]) - ord(chmin) + 1
        else: # unknown character
            ch_i = -1
        buckets[ch_i].append(word)
    return [word for bucket in buckets for word in bucket]

def word_sort(words, maxlen=None):
    """ Sort given list of words using radix sort.

    You are free to chose the method you implement, but your method
    should run in O(d*(n+m)) time (or less), where n is the number of
    words, d is the length of the longest word, and m is length of the
    alphabet.

    You can assume that the words are all in lowercase ASCII
    characters. And if ord(ch1) < ord(ch2), then the character 'ch1'
    is less than the character 'ch2' (ord is a Pyython built-in that
    returns Unicode code point of a character).  You do not need to
    sort words with non-ASCII letters correctly, but your function
    should not crash when input contains words with non-ASCII
    characters.

    Parameters
    ----------
    words:  The sequence of words to be sorted
    maxlen: The maximum length of the words (to be considered for
            sorting). For exmaple, if maxlen=3, the sorting should use
            only first three characters, even if there are words longer
            than 3 characters. If not specified, it should be set to
            the length of the longest word.
    """
    if maxlen is None:
        maxlen = max((len(word) for word in words))

    sorted_words = words
    for index in range(maxlen - 1, -1, -1):
        sorted_words = bucket_sort(sorted_words, index)

    return sorted_words

if __name__ == "__main__":

    words = open('en_wordlist.txt', 'r').read().strip().split()
    func_list = (insertion_sort, qsort, word_sort, sorted)
    n_samples = 10
    for size in (2, 4, 8, 16, 32, 64, 128, 2**10, 2**11, 2**12):
        times = [0 for _ in func_list]
        for _ in range(n_samples):
            sample = random.choices(words, k=size)
            for i, func in enumerate(func_list):
                t = time.time()
                func(sample[:])
                times[i] += time.time() - t
        for i, func in enumerate(func_list):
            print("{:>15}/{:06d}: {:.6f}".format(
                func.__name__, size, times[i]/n_samples))
        print()
