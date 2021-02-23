#!/usr/bin/env python3
""" Data Structures and Algorithms for CL III, Assignment 5
    See <https://dsacl3-2019.github.io/a45/> for detailed instructions.

    Some parts of assignment 5 has to be implemented in a4.py.

    <Please insert your name and the honor code here.>
"""

from a4 import Digraph
from scorer import Scorer, read_conllu

def mst_parse(sentence, scorer):
    """Given a sentence and scorer, return a Digraph with its MST parse.

    'sentence' is a sequence of POS tags without the artificial 'root' node.
    'scorer' is a function that returns a probability score given POS
    tags of head and dependent and the distance between them. Distance
    is defined as 'position of the head' - 'position of the dependent'.
    You are free to use the example scorer is in 'scorer.py', or write
    one yourself.
    """
    # Exercise 5.3
    g = Digraph(len(sentence) + 1, labels=['root'] + sentence)

    for i, tag1 in enumerate(sentence):
        g.set_weight(0, i + 1, scorer.score('root', tag1, 0 - i + 1))
        for j, tag2 in enumerate(sentence):
            if i == j: continue
            g.set_weight(i + 1, j + 1, scorer.score(tag1, tag2, i - j))
    t = g.mst()
    return t

def uas(gold_tree, predicted_tree):
    """Calculate the unlabeled attachment score between two trees.

    Both gold_tree and predicted_tree are instances of class Digraph.
    """
    # Exercise 5.5
    assert gold_tree.n == predicted_tree.n and gold_tree.is_tree() and predicted_tree.is_tree()
    deplist1 = set(gold_tree.edges())
    correct = 0
    for dep2 in predicted_tree.edges():
        if dep2 in deplist1:
            correct += 1
    return correct / len(deplist1)


def evaluate_parser(conllu, scorer):
    """Calculate the UAS of the parser on the sentences in the conllu file.

    The argument 'conllu' is the name of a conllu file (see test.conllu)
    for an example.
    The argument 'scorer' is to be passed to mst_parse() above.
    This function should parse all the sentences in the conllu file
    using the mst_parse() above, and calculate and return the average 
    unlabeled attachment score.
    """

    uas_sum, n = 0, 0
    for sent in read_conllu([conllu]):
        tags = [x[0] for x in sent]
        gold_tree = Digraph(len(sent), labels=tags)
        for dep, (_,head, _) in enumerate(sent[1:]):
            gold_tree.add_arc(head, dep+1)
        parse_tree = mst_parse(tags[1:], scorer)
        uas_sum += uas(gold_tree, parse_tree)
        n += 1
    return uas_sum / n

if __name__ == "__main__":
    # Feel free to use this part for quick testing (not checked).
    # You can (should for serious projects) also use pytest.

    # Here is an example of how to use the scorer with the provided
    # pre-trained model:
    s = Scorer().load('en-model.json')
    t = mst_parse(['DET', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'], s)

