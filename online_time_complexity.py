import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from sklearn.metrics import mean_squared_error as mse

np.set_printoptions(precision=3)

if __name__ == "__main__":

    data = {}
    data["bounded_time"] = []
    data["household_goods"] = []
    data["industrial_goods"] = []
    data["dependencies"] = []
    data["population"] = []
    data["free_time"] = []
    data["mse"] = []
    # data["optimality_bounded"] = []

    population = 1000000

    # memory_limit()  # limits maximun memory usage to half
    k = 0
    filename = "./data/random_%s_%s_%s_%s_%s.npz"
    for n_industrial_goods in [500, 1000, 5000, 10000, 50000]:
        for n_household_goods in [50, 100, 500, 1000, 5000]:
            for n_population in [200]:
                for n_dependencies in [500, 1000, 2000]:
                    k += 1

                    try:
                        print(n_industrial_goods, n_household_goods, n_population, n_dependencies)
                        start = time.time()

                        A_file = filename % ("A", n_industrial_goods, n_household_goods, n_population, n_dependencies)
                        I_file = filename % ("I", n_industrial_goods, n_household_goods, n_population, n_dependencies)
                        y_file = filename % ("y", n_industrial_goods, n_household_goods, n_population, n_dependencies)

                        A = sparse.load_npz(A_file)
                        I = sparse.load_npz(I_file)
                        y = np.load(y_file)["y"]

                        A = A.astype("float32")
                        I = I.astype("float32")
                        y = y.astype("float32")

                        print("it took", time.time() - start, "seconds to generate the data", n_industrial_goods,
                              n_household_goods, n_population, n_dependencies)

                        start_full = time.time()

                        X = I - A

                        print(X.shape, I.shape, y.shape)
                        print(X.indices.dtype.name)

                        x0 = np.zeros(X.shape[0], dtype="float32")
                        print("Allocated solution array")
                        x = linalg.spsolve(X.astype("double"), y.astype("double"))
                        print("Finished free solution")
                        f_time = time.time()

                        x[x < 0] = 0

                        b_time = time.time()
                        free_time = f_time - start_full
                        bounded_time = b_time - start_full
                    except:
                        x = -1
                        free_time = -1
                        bounded_time = -1
                        e = sys.exc_info()
                        print(e)

                    print("it took", bounded_time, "seconds to solve the matrix")

                    data["free_time"].append(free_time)
                    data["bounded_time"].append(bounded_time)
                    data["household_goods"].append(n_household_goods)
                    data["population"].append(n_population)
                    data["industrial_goods"].append(n_industrial_goods)
                    data["dependencies"].append(n_dependencies)
                    data["mse"].append(mse(y, X.dot(x)))

                    df = pd.DataFrame(data)
                    df.to_csv("./plot_data/times.csv")

                    print("Productive", np.all(x >= 0))
