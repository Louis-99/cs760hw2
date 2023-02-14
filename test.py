from decision_tree import *
import matplotlib.pyplot as plt

def test(tree, values):
    cnt = 0
    for v in values:
        if decision_tree_test(tree, (v[0], v[1])) == v[LABEL_INDEX]:
            cnt += 1
    
    return cnt / len(values)

def nodes_cnt(tree: LeafNode | InternalNode):
    if isinstance(tree, LeafNode):
        return 1
    return 1 + nodes_cnt(tree.then_branch) + nodes_cnt(tree.else_branch)

def get_n_err(num):
    d = read_data(f"D{num}.txt")
    t = read_data("Dtest.txt")
    tree = make_subtree(d)
    n = nodes_cnt(tree)
    err = 1 - test(tree, t)
    print(f"num: {num} n: {n}, err: {err}")
    return n, err

plt.rcParams['text.usetex'] = True

def main():
    n = [0]*5
    err = [0]*5
    n[0], err[0] = get_n_err(32)
    n[1], err[1] = get_n_err(128)
    n[2], err[2] = get_n_err(512)
    n[3], err[3] = get_n_err(2048)
    n[4], err[4] = get_n_err(8192)
    fig, ax = plt.subplots()
    ax.plot(n, err)
    ax.set_xlabel("n")
    ax.set_ylabel(r"err_{n}")
    fig.savefig("my_tree.png")


if __name__== '__main__':
    main()