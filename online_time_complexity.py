import numpy as np
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from utils import full_matrix
from scipy import sparse
from scipy.sparse import linalg
import time
from generate_matrices import get_random_matrices
import pandas as pd
import sys
import resource
from scipy.sparse.linalg import lsmr
from solver import trf_linear as local_trf_linear
from scipy.optimize._lsq.trf_linear import trf_linear


np.set_printoptions(precision=3)


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


if __name__ == "__main__":

    #time = []

    data = {}
    data["bounded_time"] = []
    data["household_goods"] = []
    data["industrial_goods"] = []
    data["dependencies"] = []
    data["population"] = []
    data["free_time"] = []
    data["mse"] = []
    #data["optimality_bounded"] = []

    population = 1000000

    #memory_limit()  # limits maximun memory usage to half
    k  = 0
    filename = "./data/random_%s_%s_%s_%s_%s.npz"
    for n_industrial_goods in [500,1000, 5000, 10000, 50000 ]:
        for n_household_goods in [50, 100, 500,1000, 5000]:
            for n_population in [200 ]:
                for n_dependencies in [500,1000, 2000  ]:
                    k+=1

                    try:
                        print(n_industrial_goods, n_household_goods, n_population, n_dependencies)
                        start = time.time()

                        A_file = filename%("A", n_industrial_goods, n_household_goods, n_population, n_dependencies)
                        I_file = filename % ("I", n_industrial_goods, n_household_goods, n_population, n_dependencies)
                        y_file = filename % ("y", n_industrial_goods, n_household_goods, n_population, n_dependencies)

                        A = sparse.load_npz(A_file)
                        I = sparse.load_npz(I_file)
                        y = np.load(y_file)["y"]

                        A = A.astype("float32")
                        I = I.astype("float32")
                        y = y.astype("float32")

                        # A.indices = A.indices.astype(np.int64)
                        # A.indptr = A.indptr.astype(np.int64)

                        #A, I, y = get_random_matrices(n_industrial_goods, n_household_goods, n_population, n_dependencies)
                        print("it took", time.time() - start, "seconds to generate the data", n_industrial_goods, n_household_goods, n_population, n_dependencies)

                        start_full = time.time()

                        X = I - A



                        print(X.shape, I.shape, y.shape)
                        print(X.indices.dtype.name)

                        x0 = np.zeros(X.shape[0], dtype="float32")
                        print("Allocated solution array")
                        #x = lsmr(X, y,maxiter=10000000, atol = 1.0e-6, btol = 1.0e-6, conlim = 0, x0 = x0)[0]
                        x = linalg.spsolve(X.astype("double"), y.astype("double"))
                        print("Finished free solution")
                        f_time = time.time()

                        x[x < 0] = 0

                        # lb = np.array([0] * y.shape[0])
                        # ub = x * np.random.random(size=x.shape[0])
                        # ub[-n_population:] = 1
                        #
                        # x2 = trf_linear(X, y, ub, lb, ub, 0.1, "lsmr", 0.00001, 10000, 0)


                        b_time = time.time()
                        free_time =  f_time - start_full
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
                    #data["optimality_bounded"].append(x2["optimality"])

                    df = pd.DataFrame(data)
                    df.to_csv("./plot_data/times.csv")



                    print("Productive", np.all(x >= 0))
                    #print(x [x < 0])
                    #print(x)
                    #print(X.dot(x))

                    #print("MSE", ))

                    # metric.append("mse")
                    # value.append(float(total_time))
                    # iterations.append(i)

