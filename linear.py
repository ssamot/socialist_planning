from generate_matrices import y_file, A_file, I_file
from sklearn.linear_model import LinearRegression, SGDRegressor
import numpy as np
import scipy.sparse
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from utils import full_matrix
from generate_matrices import A_file, I_file, y_file, n_household_goods, n_industrial_goods, n_population
from scipy import sparse
from scipy.sparse import linalg



if __name__ == "__main__":


    production_df = pd.read_csv("data/butter_production.csv")
    demand_df = pd.read_csv("data/butter_demand.csv")

    # overproduction = 2
    # matrix = full_matrix(production_df.copy(), demand_df, overproduction)
    #
    # A = matrix[:, :-1]
    # I = np.eye(A.shape[0])
    # y = matrix[:, -1]


    A = sparse.load_npz(A_file)
    I = sparse.load_npz(I_file)
    y = np.load(y_file)["y"]

    X = I - A
    print(X.shape, I.shape, y.shape)
    #clf = VWRegressor()
    #clf = SGDRegressor(fit_intercept=False, verbose=True, eta0 = 0.001 )
    #clf = LinearRegression(fit_intercept=False, copy_X=False)
    print(type(X), type(y), X.shape, y.shape)
    #X = np.array(X.todense(), dtype = "float")
    #y = np.array(y, dtype="float")
    #x = np.linalg.solve(X,y)
    x = linalg.spsolve(X.astype("double"), y)
    #x = linalg.lsqr(X,y)[0]
    #print(x)
    #clf.fit(X,y)
    #print(clf.get_coefs())
    #print(clf.coef_[:4])
    #print("sfsdfs")
    #print(x)
    print(X.dot(x))
    #print(y)

    print(mse(y, X.dot(x)))


