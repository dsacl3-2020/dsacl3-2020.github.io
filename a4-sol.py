#!/usr/bin/env python3
""" Data Structures and Algorithms for CL III, Assignment 4 & 5
    See <https://dsacl3-2019.github.io/a45/> for detailed instructions.

    This file mainly contains parts related to both assignment 4 and 5.

    <Please insert your name and the honor code here.>
"""
import numpy as np
import sys

class Digraph:
    def __init__(self, n=0, labels=None):
        assert n or labels
        self.labels = None
        self.n = n
        if labels:
            self.n = len(labels)
            self.labels = labels
        self.w = np.zeros(shape=(self.n, self.n))

    def nodes(self):
        return range(self.n)

    def edges(self, return_weight=False):
        for i in range(self.n):
            for j in range(self.n):
                if self.w[i, j] != 0:
                    if return_weight:
                        yield (i, j, self.w[i, j])
                    else:
                        yield (i, j)

    def get_weight(self, u, v):
        return self.w[u, v]

    def set_weight(self, u, v, weight):
        self.add_arc(u, v, weight)

    def add_arc(self, source, target, weight=1.0):
        self.w[source, target] = weight

    def in_edges(self, v):
        return [x for x in range(self.n) if self.w[x, v] != 0.0]

    def in_degree(self, v):
        return len(self.in_edges(v))

    def out_edges(self, v):
        return [x for x in range(self.n) if self.w[v, x] != 0.0]

    def out_degree(self, v):
        return len(self.out_edges(v))

    def dfs(self, start=0, reverse=False):
        stack = [start]
        visited = {start: None}
        while stack:
            node = stack.pop()
            if reverse:
                to_visit = self.in_edges(node)
            else:
                to_visit = self.out_edges(node)
            for child in to_visit:
                if child not in visited:
                    visited[child] = node
                    stack.append(child)
        return visited

    def get_strongly_connected(self, node):
        dfs_forward = set(self.dfs(node))
        dfs_backward = set(self.dfs(node, reverse=True))
        return dfs_forward & dfs_backward

    def is_strongly_connected(self):
        # There is some repetition here (we could use get_strongly_connected()
        # but this is slightly more efficient since it returns
        # immediately if the forward dfs does not visit all nodes
        dfs_tree = self.dfs(0)
        if set(range(self.n)) != set(dfs_tree):
            return False
        dfs_tree = self.dfs(0, reverse=True)
        if set(range(self.n)) != set(dfs_tree):
            return False
        return True

    def _find_cycle(self, start=0):
        stack = [start]
        visited = {start: None}
        #        cycles = []
        while stack:
            node = stack.pop()
            for child in self.out_edges(node):
                if child not in visited:
                    visited[child] = node
                    stack.append(child)
                else:  # we have a back or cross edge
                    # find the path from start to currrent node
                    # this could be done more efficiently
                    path = [node]
                    curr = node
                    while curr != start:
                        curr = visited[curr]
                        path.append(curr)
                    if child in path:  # back edge
                        i = path.index(child)
                        return reversed(path[:i + 1])
        #                       cycles.append(list(reversed(path[:i+1])))
        #        return cycles
        return None

    def find_cycle(self):
        checked = set()
        for node in self.nodes():
            if node in checked: continue
            checked.update(self.get_strongly_connected(node))
            cycle_n = self._find_cycle(node)
            if cycle_n:
                return cycle_n
        return []

    def is_cyclic(self):
        return len(self.find_cycle()) != 0
        # alternatively return set(self.sort_topo()) == set(self.nodes())

    def is_tree(self, root=0):
        if self.is_cyclic():  # cyclic cannot be a tree
            return False
        if len(self.in_edges(root)) != 0:  # root has an incoming edge
            return False
        for node in self.nodes():
            if node != root and len(self.in_edges(node)) != 1:
                # multiple parents
                return False
        if set(self.dfs(root)) != set(self.nodes()):
            # not connected
            return False
        return True

    def sort_topo(self):
        ready = []
        topo = []
        incount = [len(self.in_edges(v)) for v in self.nodes()]
        for v in self.nodes():
            if incount[v] == 0:
                ready.append(v)

        while len(ready) != 0:
            u = ready.pop()
            topo.append(u)
            for v in self.out_edges(u):
                incount[v] -= 1
                if incount[v] == 0:
                    ready.append(v)
        return topo

    def to_dot(self, filename=None, use_labels=False):
        if filename:
            fp = open(filename, 'wt')
        else:
            fp = sys.stdout
        if use_labels:
            labels = ["".join((l, str(i))) for i, l in enumerate(self.labels)]
        else:
            labels = list(range(self.n))
        print(r"digraph {rankdir = LR;node[shape=circle];", file=fp)
        for v1 in range(self.n):
            for v2 in range(self.n):
                if self.w[v1, v2] != 0.0:
                    l1, l2 = v1, v2
                    print("  {} -> {} [label=\"{:0.4f}\"];".format(
                        labels[v1], labels[v2], self.w[v1, v2]), file=fp)

        print(r"}", file=fp)
        if filename: fp.close()

    def mst(self, root=0):
        print(self.in_degree(root))
        assert self.in_degree(root) == 0
        mst = Digraph(self.n, labels=self.labels)
        for v in self.nodes():
            if v == root: continue
            maxw, maxu = 0.0, None
            for u in self.in_edges(v):
                w = self.get_weight(u, v)
                if w > maxw:
                    maxw, maxu = w, u
            mst.add_arc(maxu, v, maxw)
        cycle = list(mst.find_cycle())
        removed = set()
        while len(cycle):
            minloss, bestu, bestv, oldp = float('inf'), None, None, None
            for v in cycle:
                currp = mst.in_edges(v)[0]
                currw = mst.get_weight(currp, v)
                for u in self.in_edges(v):
                    if u in cycle: continue
                    if (u, v) in removed: continue
                    uw = self.get_weight(u, v)
                    if currw - uw < minloss:
                        minloss = currw - uw
                        bestu, bestv, oldp = u, v, currp
            mst.set_weight(oldp, bestv, 0.0)
            removed.add((oldp, bestv))
            mst.set_weight(bestu, bestv, self.get_weight(bestu, bestv))
            cycle = list(mst.find_cycle())
        return mst

    def is_projective(self, root=0):
        if not self.is_tree(root):
            return False
        for edge1 in self.edges():
            for edge2 in self.edges():
                s1, e1 = sorted(edge1)  # the span of first dependency
                s2, e2 = sorted(edge2)  # the span of second dependency
                if s1 < s2 < e1 < e2 or s2 < s1 < e2 < e1: # first e
                    return False
        return True


if __name__ == "__main__":
    # Feel free to use this part for quick testing (not checked).
    # You can (should for serious projects) also use pytest.
    labels = ["A", "B", "C", "D", "E"]
    di3 = Digraph(labels=labels)
    di3.set_weight(0, 1, 1.0)
    di3.set_weight(1, 2, 1.5)
    di3.set_weight(2, 3, 2.5)
    di3.set_weight(2, 4, 3.0)
    di3.set_weight(4, 0, 6.0)
