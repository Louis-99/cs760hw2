# Author: Yunzhao Liu
# environment: Python 3.10.9
from __future__ import annotations
from typing import *
import math
from functools import cmp_to_key
import numpy as np

LABEL_INDEX = 2

class LeafNode:
    def __init__(self, label: int):
        self.label = label
    
    def __repr__(self) -> str:
        return f"(LABEL: {self.label})"

class InternalNode:
    def __init__(self, condition: Tuple[int, float], then_branch: InternalNode | LeafNode, else_branch: InternalNode | LeafNode):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch
    
    def __repr__(self) -> str:
        return f"(COND: ({self.condition[0]+1}, {self.condition[1]}), THEN: {self.then_branch}, ELSE: {self.else_branch})"



def make_subtree(training_set: list[tuple[float, float, bool]]) -> InternalNode | LeafNode:
    split_set = determine_candidate_splits(training_set)
    best_split = find_best_split(training_set, split_set)
    if best_split is None:
        return LeafNode(determine_label(training_set))
    else:
        (dim, thres) = best_split
        then_set = [x for x in training_set if x[dim] >= thres]
        else_set = [x for x in training_set if x[dim] < thres]
        return InternalNode(best_split, make_subtree(then_set), make_subtree(else_set))


def determine_candidate_splits(training_set: list[tuple[float, float, bool]]) -> list[tuple[int, float]]:
    res : list[tuple[int, float]] = []

    min0 = min(x[0] for x in training_set)
    min1 = min(x[1] for x in training_set)

    doubleState = False

    cmp_index = 0
    def cmp(lhs, rhs):
        if lhs[cmp_index] != rhs[cmp_index]:
            return lhs[cmp_index]-rhs[cmp_index]
        else:
            return lhs[LABEL_INDEX]-rhs[LABEL_INDEX]

    set0 = sorted(training_set, key=cmp_to_key(cmp))
    for i in range(1, len(set0)):
        if set0[i][LABEL_INDEX] != set0[i-1][LABEL_INDEX] or doubleState:
            if set0[i][0] > min0:
                res.append((0, set0[i][0]))
        
        if set0[i][LABEL_INDEX] != set0[i-1][LABEL_INDEX]:
            doubleState = (set0[i][0] == set0[i-1][0])
    
    cmp_index = 1
    doubleState = False
    set1 = sorted(training_set, key=cmp_to_key(cmp))
    for i in range(1, len(training_set)):
        if set1[i][LABEL_INDEX] != set1[i-1][LABEL_INDEX] or doubleState:
            if set1[i][1] > min1:
                res.append((1, set1[i][1]))
        
        if set0[i][LABEL_INDEX] != set0[i-1][LABEL_INDEX]:
            doubleState = (set0[i][0] == set0[i-1][0])

    return res

# if returns None, then sopping criteria meets
def find_best_split(training_set: list[tuple[float, float, bool]], split_set: list[tuple[int, float]]) -> tuple[int, float] | None:
    if len(training_set) == 0:
        return None
    rank = []
    for i, (dim, c) in enumerate(split_set):
        gr = gain_ratio(training_set, dim, c)
        if gr is not None:
            rank.append((gr, i))
    if len(rank) == 0:
        return None
    max_split_index = max(rank, key=lambda x: x[0])
    if max_split_index[0] == 0:
        return None
    
    return split_set[max_split_index[1]]

def determine_label(training_set: list[tuple[float, float, bool]]) -> int:
    cnt1 = sum(1 for x in training_set if x[LABEL_INDEX])
    if cnt1 >= len(training_set)/2:
        return 1
    else:
        return 0


def read_data(filename: str) -> list[tuple[float, float, bool]]:
    res = []
    with open(filename, "r") as file:
        while True:
            line = file.readline()
            if line == "":
                break
            tokens = line.split(' ')
            data = (float(tokens[0]), float(tokens[1]), bool(int(tokens[LABEL_INDEX])))
            res.append(data)
    return res

def info_gain(data: list[tuple[float,float,int]], dim: int, thres: float) -> float | None:
    # Y is D, X is S
    H_Y = entropy(data)
    P_X0 = sum(1 for x in data if x[dim] < thres) / len(data)
    P_X1 = 1-P_X0
    H_Y_X0 = entropy([x for x in data if x[dim] < thres])
    H_Y_X1 = entropy([x for x in data if x[dim] >= thres])
    if H_Y_X0 is None or H_Y_X1 is None:
        return None
    H_Y_X = P_X0 * H_Y_X0 + P_X1 * H_Y_X1
    I_Y_X = H_Y - H_Y_X
    return I_Y_X

# return None if entropy of the candidate split is zero
def gain_ratio(data: list[tuple[float,float,int]], dim: int, thres: float) -> float | None:
    # Y is D, X is S
    H_Y = entropy(data)
    P_X0 = sum(1 for x in data if x[dim] < thres) / len(data)
    P_X1 = 1-P_X0
    H_Y_X0 = entropy([x for x in data if x[dim] < thres])
    H_Y_X1 = entropy([x for x in data if x[dim] >= thres])
    if H_Y_X0 is None or H_Y_X1 is None:
        return None
    H_Y_X = P_X0 * H_Y_X0 + P_X1 * H_Y_X1
    I_Y_X = H_Y - H_Y_X
    H_X = - mullog(P_X0) - mullog(P_X1)
    if H_X == 0: 
        return None
    return I_Y_X / H_X


def entropy(data: list[tuple[float, float, bool]]) -> float | None:
    cnt = len(data)
    if cnt == 0:
        return None
    cnt1 = sum(1 for x in data if x[LABEL_INDEX])
    cnt0 = cnt - cnt1
    p0 = cnt0/cnt
    p1 = cnt1/cnt
    return - mullog(p0) - mullog(p1)

def mullog(v: float) -> float:
    if v < 1e-9:
        return 0
    else:
        return v * math.log2(v)

def decision_tree_test(tree: LeafNode | InternalNode, value: Tuple[float, float]):
    if isinstance(tree, LeafNode):
        return tree.label
    if value[tree.condition[0]] >= tree.condition[1]:
        return decision_tree_test(tree.then_branch, value)
    else:
        return decision_tree_test(tree.else_branch, value)

if __name__ == '__main__':
    d = read_data("D2.txt")
    c = determine_candidate_splits(d)
    for dim, thres in c:
        ig = gain_ratio(d, dim, thres)
        print(f"cut:({dim+1}, {thres}) gain ratio: {ig}")
        
    tree = make_subtree(d)
    print(tree)
