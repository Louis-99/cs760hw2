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
    n = n*2-1
    print(f"n: {n}, err: {err}")
    return n, err


# plt.rcParams['text.usetex'] = True

if __name__ == '__main__':
    ns = [32, 128, 512, 2048, 8192]
    nerr = [test(f"D{n}.txt") for n in ns]
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in nerr], [x[1] for x in nerr])
    ax.set_xlabel("n")
    ax.set_ylabel(r"err_{n}")
    fig.savefig("use_sklearn_tree.png")

# n: 11, err: 0.13716814159292035
# n: 27, err: 0.06360619469026549
# n: 55, err: 0.043694690265486724
# n: 107, err: 0.02765486725663717
# n: 249, err: 0.010508849557522125