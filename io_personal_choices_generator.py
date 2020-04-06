import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from utils import full_matrix
from metrics import calculate_percentages, humanity
from tqdm import tqdm

from nn import nn
from utils import batch_generator


def measure_humanity(X,y):
    pass

def measure_modality(X,y):
    pass








if __name__ == "__main__":
    production_df = pd.read_csv("data/butter_production.csv")
    demand_df = pd.read_csv("data/butter_demand.csv")

    overproduction = 2
    matrix = full_matrix(production_df.copy(), demand_df, overproduction)


    A = matrix[:, :-1]
    I = np.eye(A.shape[0])
    y = matrix[:, -1]
    ones_full = np.ones(shape=(A.shape[0], 3))


    model, X_model = nn(len(A.T))

    metric = []
    value = []
    iterations = []

    max_iterations = 20000
    with tqdm(total=max_iterations) as pbar:
        #print(len(demand_df)); exit()
        for i, [A_batch, I_batch, y_batch] in enumerate(batch_generator([A,I, y], batch_size=1, split = len(A) - len(demand_df))):
            ones = np.ones(shape=(A_batch.shape[0], 3))

            model.train_on_batch([A_batch, I_batch,  ones], y_batch)

            MSE = mse(y, model.predict([A,I, ones_full]))
            metric.append("mse")
            value.append(float(MSE))
            iterations.append(i)

            hu = humanity(A, I,len(A) - len(demand_df), 2, model)
            metric.append("$\mathcal{HU}_p$")
            value.append(float(hu))
            iterations.append(i)
            pbar.set_postfix({"hu":"%0.3f" % hu, "mse":"%0.3f" % MSE})
            pbar.update()

            if(i > max_iterations):
                break

    print(len(value), len(metric), len(iterations))
    df = pd.DataFrame({"metric":metric, "value":value, "iteration":iterations})
    df.to_csv("./plot_data/butter.csv")
        #x = X_model.predict(np.ones(shape = (1, 3)))
        #hu_1 = calculate_percentages(I - A, x production_df, demand_df, model)

