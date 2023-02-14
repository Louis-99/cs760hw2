from decision_tree import *

import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import numpy as np
def plot_data_set(ax: plt.Axes, data):
    ax.scatter(
        list(x[0] for x in data if x[LABEL_INDEX]),
        list(x[1] for x in data if x[LABEL_INDEX]),
        c = 'r',
    )
    ax.scatter(
        list(x[0] for x in data if not x[LABEL_INDEX]),
        list(x[1] for x in data if not x[LABEL_INDEX]),
        c = 'b',
    )

def plot_tree_area(ax: plt.Axes, tree: LeafNode | InternalNode, xmin, xmax, ymin, ymax):
    if isinstance(tree, LeafNode):
        if tree.label == 0:
            color = 'y'
        else:
            color = 'c'
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, linewidth=0))
        return
    
    dim, thres = tree.condition
    if dim == 0:
        thres = max(min(thres, xmax), xmin)
        plot_tree_area(ax, tree.then_branch, thres, xmax, ymin, ymax)
        plot_tree_area(ax, tree.else_branch, xmin, thres, ymin, ymax)
    else:
        thres = max(min(thres, ymax), ymin)
        plot_tree_area(ax, tree.then_branch, xmin, xmax, thres, ymax)
        plot_tree_area(ax, tree.else_branch, xmin, xmax, ymin, thres)

def plot_dataset_and_tree(tree_ds_name, test_ds_name, output_name):
    tree_ds = read_data(tree_ds_name)
    test_ds = read_data(test_ds_name)
    t = make_subtree(tree_ds)
    print(t)
    fig = plt.figure(figsize=(7,7), dpi=100)
    ax = fig.subplots()
    plot_tree_area(ax, t, -1.5, 1.5, -1.5, 1.5)
    plot_data_set(ax, test_ds)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.savefig(output_name)
    return fig 


def main_for_prob7():
    plot_dataset_and_tree("D32.txt", "Dtest.txt", "test32.png")
    plt.show()
    plot_dataset_and_tree("D128.txt", "Dtest.txt", "test128.png")
    plt.show()
    plot_dataset_and_tree("D512.txt", "Dtest.txt", "test512.png")
    plt.show()
    plot_dataset_and_tree("D2048.txt", "Dtest.txt", "test2048.png")
    plt.show()
    plot_dataset_and_tree("D8192.txt", "Dtest.txt", "test8192.png")
    plt.show()

def main_for_prob6():
    d1 = read_data("D1.txt")
    fig, ax = plt.subplots()
    plot_data_set(ax, d1)
    ax.set_xlabel("dimension 0")
    ax.set_ylabel("dimension 1")
    ax.set_title("data point for D1")
    fig.savefig("D1datapoint.png")

    fig, ax = plt.subplots()
    t = make_subtree(d1)
    plot_tree_area(ax, t, 0, 1, 0, 1)
    ax.set_xlabel("dimension 0")
    ax.set_ylabel("dimension 1")
    ax.set_title("Boundary D1")
    fig.savefig("D1boundary.png")

    d2 = read_data("D2.txt")
    fig, ax = plt.subplots()
    plot_data_set(ax, d2)
    ax.set_xlabel("dimension 0")
    ax.set_ylabel("dimension 1")
    ax.set_title("data point for D2")
    fig.savefig("D2datapoint.png")

    fig, ax = plt.subplots()
    t = make_subtree(d2)
    plot_tree_area(ax, t, 0, 1, 0, 1)
    ax.set_xlabel("dimension 0")
    ax.set_ylabel("dimension 1")
    ax.set_title("Boundary D2")
    fig.savefig("D2boundary.png")

if __name__ == '__main__':
    main_for_prob6()
