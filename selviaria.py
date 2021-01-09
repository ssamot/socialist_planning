import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import pandas as pd
from utils import full_matrix
from scipy.sparse import linalg
from metrics import humanity
from scipy.sparse.linalg import lsmr
from scipy.optimize import minimize_scalar
from coeffs import get_io_approximations
from argparse import ArgumentParser


def get_coeff_matrix(A, I, xtm, io_coeff_functions):
    for (i, j, f) in io_coeff_functions:
        A[i, j] = f(xtm[j])
    return I - A


def solve_eq(X, y, io_coeff_functions, A, I, xtm):
    if (io_coeff_functions is None):
        x = lsmr(X.astype("double"), y.astype("double"), maxiter=100000, atol=0, btol=0, conlim=0)[0]
    else:
        x = xtm + 0.001
        for i in range(3):
            X = get_coeff_matrix(A, I, x, io_coeff_functions)
            x = linalg.spsolve(X.astype("double"), y.astype("double"))
    return x


def solve(production_df, demand_df, ub, io_coeff_functions, citizens):
    v, production_df = full_matrix(production_df.copy(), demand_df, 1)
    n_profiles = demand_df.values.shape[0]

    A = v[:, :-1]
    I = np.eye(A.shape[0])
    y = v[:, -1]
    X = I - A
    y[-n_profiles:] = citizens

    if (ub is None):
        x = solve_eq(X, y, io_coeff_functions, A, I, y)
    else:
        x = solve_eq(X, y, io_coeff_functions, A, I, y)
        ub_f = x.copy()
        ub_f[:ub.shape[0]] = ub[:]

        def fun(d):
            x = solve_eq(X, y / d, io_coeff_functions, A, I, y / d)
            r = ub_f - x
            r[r > 0] = 0
            r = -r
            # print(r, r.sum(), d,  "test")
            if (r.sum() > 0.00001):
                return r.sum()
            return (-np.sum(x[-n_profiles:]))

        mult = minimize_scalar(fun, bounds=[1, 10000], method="bounded")["x"]

        x = solve_eq(X, y / mult, io_coeff_functions, A, I, y / mult)
    X = get_coeff_matrix(A, I, y, io_coeff_functions)
    hu = humanity(A, I, y, len(A) - len(demand_df), len(demand_df.T), x, s=False, sparse=False)
    return x, y, np.array(X.dot(x), dtype="double"), hu


def generate_data(noise, csv):
    production_df = pd.read_csv("data/selviaria.csv")
    demand_df = pd.read_csv("data/selveria_profiles.csv")

    n_production_items = 6
    n_ticks = 1000

    a, b = get_io_approximations()
    io_coeff_functions = [(1, 1, a), (1, 1, b)]

    data = {}
    data["tick"] = []
    data["$\mathcal{HU}_p$"] = []
    data["investment"] = []
    data["externalities"] = []

    for investment in (0.51, 0.52, 0.55, 0.60):
        hu = 0
        total_inventory = np.zeros(shape=n_production_items)
        total_inventory[0] = 2
        total_inventory[1] = 5
        total_inventory[2] = 100000
        # labour bounds
        total_inventory[3] = 100000
        total_inventory[4] = 700000
        total_inventory[5] = 700000

        for i in range(n_ticks):
            if (hu > 0.999):
                continue
            x, y, y_hat, hu = solve(production_df, demand_df, total_inventory, io_coeff_functions, citizens=[800, 500])

            total_inventory[1] += x[1]
            total_inventory[0] -= x[0] * (1 - investment)
            total_inventory[0] += x[0] * investment

            if (noise):
                if (np.random.random() < 0.1):
                    total_inventory[1] = total_inventory[1] * 0.8
                    total_inventory[0] = total_inventory[0] * 0.8

            data["$\mathcal{HU}_p$"].append(hu)
            data["investment"].append(investment)
            data["tick"].append(i)
            print(i, investment, hu)

    df = pd.DataFrame(data)
    df.to_csv(csv)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-noisy', action='store_true')

    args = parser.parse_args()
    if(args.noisy):
        generate_data(True, csv="./plot_data/selviaria-noise.csv")
    else:
        generate_data(False, csv="./plot_data/selviaria.csv")
