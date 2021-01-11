#!/usr/bin/env python3

""" Data Structures and Algorithms for CL III, Assignment 2
    See <https://dsacl3-2019.github.io/a2/> for detailed instructions.
    <Please insert your name and the honor code here.>
"""

import numpy as np
from collections import Counter


class WordScore:
    """ A class for calculating word/letter probabilities.
        Please see the assignment page for descripiton.
    """

    def __init__(self):
        """ Any initialization you need can go in here.
        """
        self.words = None
        self.letters = None
        self.a = None
        self.nwords = None
        self.nletters = None

    def train(self, corpus):
        """ Given a set of sentences (strings) calculate/update  frequencies.
        You are free to organize the information you collect here the
        way you like. However, an instance of this class should be able to
        calculate the probability of a word (as described in the
        assignment sheet) based on the information collected here.
        Parameters:
        corpus   A list of list of strings. Where each string is a word.
                 For example a two-sentence corpus "a kitty" and "nice bunny"
                 should be given as [['a', 'kitty'], ['nice', 'bunny']].
        Return value: None
        """
        corpus = [w for s in corpus for w in s]
        self.words = Counter(corpus)
        self.letters = Counter()
        for word in corpus:
            self.letters.update(word)
        self.nwords = sum(self.words.values())
        self.nletters = sum(self.letters.values())
        # count(1) returns number of times 1 appears in the list
        self.a = list(self.words.values()).count(1) / self.nwords

    def score(self, word):
        """ Given a word, return its (log) probability.
        Given a word, check if it is a known word, and calculate and
        return its probability accordingly. See the assignment sheet
        for details. Return value as both probability and log
        probability are acceptable from this function. But you are
        strongly recommended to work with log probabilities.
        Parameters:
        word  A possible word (a string).
        Return value: Either probability or log probability of the word.
        """
        assert self.words is not None, "You need to train first."
        if word in self.words:
            return np.log(1 - self.a) + np.log(self.words[word] / self.nwords)
        else:
            logprob = 0
            for l in word:
                # this calculates add+1-smoothed probabilities to make
                # sure that unknown letters are treated correctly.
                # not required, using simply the relative
                # frequency is sufficient.
                logprob += np.log(self.letters.get(l, 1) /
                                  (self.nletters + len(self.letters)))
            return np.log(self.a) + logprob


    '''def score(self, word):
        assert self.words is not None, "You need to train first."
        if word in self.words:
            return (1 - self.a) * (self.words[word] / self.nwords)
        else:
            prob = 1
            for l in word:
                prob *= (self.letters[l] /self.nletters)
            return self.a * prob'''


def segment_r(seq):
    if len(seq) == 1:
        yield [seq]
    else:
        for seg in segment_r(seq[1:]):
            yield [seq[0]] + seg
            yield [seq[0] + seg[0]] + seg[1:]


def segment_bf(seq, score_func):
    """ Calculate the optimal segmentation of the given sequence (a string).
    This function generates all segmentation of a string in a
    brute-force manner and finds the segmentation with the highest
    score. You are even more strongly recommended to use log
    probabilities here.
    Parameters:
    seq         A sequence (string) without any space characters.
    score_func  A function that returns a word score for a given
                string. Intended use is WordScore.score(), but code
                should work with any function that returns a higher
                score for a better word.
    Return value: the best segmentation and its score (a tuple)

    """
    maxsc = float('-inf')     # necessary for log prob
    bestseg = None
    for seg in segment_r(seq):
        score = sum([score_func(w) for w in seg]) # np.prod([score_func(w) for w in seg])
        if score > maxsc:
            bestseg = seg
            maxsc = score
    return maxsc, bestseg


def segment_dp(seq, score_func, seg_scores=None):
    """ Calculate the optimal segmentation of the given sequence (a string).
    This is the dynamic programming version of segment_bf() above. The
    API should be the same, but you can use additional parameters for
    keeping static/persistent information (not the best way, but
    works).
    For a sketch, please see the assignment sheet.
    """
    if seg_scores is None:
        seg_scores = dict()
    if seq in seg_scores:
        return seg_scores[seq]
    if len(seq) == 1:
        sc = score_func(seq)
        seg_scores[seq] = sc, [seq]
        return sc, [seq]
    else:
        bestseg = [seq]
        maxsc = score_func(seq)
        for i in range(1, len(seq)):
            sc1, seg1 = segment_dp(seq[:i], score_func, seg_scores)
            sc2, seg2 = segment_dp(seq[i:], score_func, seg_scores)
            if (sc1 + sc2) > maxsc: # if (sc1 * sc2) > maxsc:
                maxsc = sc1 + sc2   # maxsc = sc1 * sc2
                bestseg = seg1 + seg2
        seg_scores[seq] = maxsc, bestseg
        return maxsc, bestseg


def evaluate(gold, pred):
    """Return precision and recall for a test set.
    Given gold standard and predicted segmentations, return precision
    and recall values. Please see the assignment sheet for more
    information.
    Parameters:
    gold     A list of list of strings where each string is a word.
             For example a two-sentence corpus "a kitty" and "nice bunny"
             should be given as [['a', 'kitty'], ['nice', 'bunny']].
    pred     Same format/shape as gold.
    Return value: precision and recall (a tuple)
    """
    assert len(gold) == len(pred)

    tp, fp, fn = 0, 0, 0
    for i, goldseg in enumerate(gold):
        predseg = pred[i]
        # uses word lengths to calculate boundary positions
        gboundary = set(np.cumsum([len(x) for x in goldseg[:-1]])) # seg[:-1] so we don't count end of word as boundary
        pboundary = set(np.cumsum([len(x) for x in predseg[:-1]]))
        # use set theory to find out which boundary positions appear only in gold(false negative)
        # or only in pred(false positive) or in both(true positive)
        tp += len(gboundary & pboundary)
        fp += len(pboundary - gboundary)
        fn += len(gboundary - pboundary)
    if tp == 0:
        prec, recall = 0.0, 0.0
    else:
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
    return prec, recall


if __name__ == "__main__":
    # Your assignment will be graded based on the correct
    # implementation of the functions above.
    #
    # You can use this for calling the above functions
    # properly and testing your code. Proper unit testing using
    # 'pytest' is also welcome (but not required).

    sentences = open('childes.text').read().strip().split('\n')
    testsent = sentences[:len(sentences) // 2]
    trainsent = sentences[len(sentences) // 2:]
    test = [s.split() for s in testsent]
    train = [s.split() for s in trainsent]
    ws = WordScore()
    ws.train(train)

    pred = []
    for sent in testsent:
        sc, seg = segment_dp(sent.replace(' ', ''), ws.score)
        pred.append(seg)
    print(evaluate(test, pred))