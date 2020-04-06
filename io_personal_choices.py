import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import full_matrix
from metrics import calculate_percentages

from nn import nn








if __name__ == "__main__":

    devisor = 1

    production_df = pd.read_csv("data/butter_production.csv")
    demand_df = pd.read_csv("data/butter_demand.csv")

    matrix = full_matrix(production_df.copy(), demand_df, 2)
    print(matrix)


    A = matrix[:, :-1]/devisor
    I = np.eye(A.shape[0])/devisor
    y = matrix[:, -1]/devisor
    ones = np.ones(shape = (A.shape[0],3) )
    one = np.ones(shape = (1, 3))
    #X = I - A


    model, X_model = nn(len(A.T))
    model.fit([A, I,  ones], y, epochs=1000, batch_size=30, shuffle=True, verbose=True)

    print("MSE", mean_squared_error(y, model.predict([A,I, ones])))

    x = X_model.predict(one)
    l = dict(zip(production_df.columns[1:-1], x.T))

    solution = pd.DataFrame(l)
    print(solution)
    #print("A",x.dot(A))
    calculate_percentages(I-A, x, production_df, demand_df, model)


    print("actual output", model.predict([A, I,  ones]))
