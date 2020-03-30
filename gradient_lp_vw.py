from generate_matrices import y_file, A_file, I_file
from sklearn.linear_model import LinearRegression, SGDRegressor
import numpy as np
import scipy.sparse
from sklearn.metrics import mean_squared_error as mse
import pandas as pd


if __name__ == "__main__":
    # A = scipy.sparse.load_npz(A_file)
    # I = scipy.sparse.load_npz(I_file)
    # y = np.load(y_file)["y"]

    dep_df = pd.read_csv("data/butter2.csv")

    matrix = dep_df.values[:, 1:]
    A = matrix[:, :-1]
    I = np.eye(A.shape[0])
    y = matrix[:, -1]

    X = I- A
    print(X.shape, I.shape, y.shape)
    #clf = VWRegressor()
    clf = SGDRegressor(fit_intercept=False, verbose=True )
    #clf = LinearRegression(fit_intercept=False, copy_X=False)
    clf.fit(X,y)
    #print(clf.get_coefs())
    print(clf.coef_)
    print(X)

    print(mse(y, clf.predict(X)))


