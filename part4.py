import numpy as np
import scipy
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

def main1():
    print("without noise")
    a = 0
    b = 1
    x = np.linspace(a, b, 99)
    train_x = np.linspace(a, b, 10)
    train_y = np.sin(train_x)

    ploy_coef = lagrange(train_x, train_y).coef[::-1]
    ploy = Polynomial(ploy_coef)

    train_error = np.linalg.norm(train_y - ploy(train_x), 2)/len(train_x)
    test_error = np.linalg.norm(np.sin(x) - ploy(x))/len(x)
    

    print(f"train_error = {train_error}")
    print(f"test_error = {test_error}")

def main2():
    print("with noise")
    a = 0
    b = 1
    x = np.linspace(a, b, 99)
    train_x = np.linspace(a, b, 10) + np.random.normal(0, 1, 10)
    train_y = np.sin(np.linspace(a, b, 10))

    ploy_coef = lagrange(train_x, train_y).coef[::-1]
    ploy = Polynomial(ploy_coef)

    train_error = np.linalg.norm(train_y - ploy(train_x), 2)/len(train_x)
    test_error = np.linalg.norm(np.sin(x) - ploy(x))/len(x)
    

    print(f"train_error = {train_error}")
    print(f"test_error = {test_error}")


if __name__ == '__main__':
    main1()
    main2()