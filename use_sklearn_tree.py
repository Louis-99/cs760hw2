from sklearn.tree import DecisionTreeClassifier
from decision_tree import read_data
import matplotlib.pyplot as plt

def test(ds_name):
    ds = read_data(ds_name)
    test_ds = read_data("Dtest.txt")
    dtc = DecisionTreeClassifier()
    dtc.fit(
        [[x[0], x[1]] for x in ds],
        [int(x[2]) for x in ds],
    )
    ret_y = dtc.predict(
        [[x[0], x[1]] for x in test_ds],
    )
    cnt = sum(int(test_ds[i][2]) != ret_y[i] for i in range(1, len(test_ds)))
    err = (cnt/len(test_ds))
    n = dtc.get_n_leaves()
    print(f"n: {n}, err: {err}")
    return n, err


plt.rcParams['text.usetex'] = True

if __name__ == '__main__':
    ns = [32, 128, 512, 2048, 8192]
    nerr = [test(f"D{n}.txt") for n in ns]
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in nerr], [x[1] for x in nerr])
    ax.set_xlabel("n")
    ax.set_ylabel(r"err_{n}")
    fig.savefig("use_sklearn_tree.png")

