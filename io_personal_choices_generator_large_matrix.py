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
from scipy import sparse
from generate_matrices import A_file, I_file, y_file, n_household_goods, n_industrial_goods, n_population





if __name__ == "__main__":
    A = sparse.load_npz(A_file)
    I = sparse.load_npz(I_file)
    y = np.load(y_file)["y"]

    ones_full = np.ones(shape=(A.shape[0], 3))


    model, X_model = nn(A.shape[1])

    metric = []
    value = []
    iterations = []

    max_iterations = 20000
    with tqdm(total=max_iterations) as pbar:
        for i, [A_batch, I_batch, y_batch] in enumerate(batch_generator([A,I, y], batch_size=2, split = n_household_goods + n_industrial_goods)):
            #print(A_batch.shape)
            ones = np.ones(shape=(A_batch.shape[0], 3))

            model.train_on_batch([A_batch, I_batch,  ones], y_batch)

            MSE = mse(y, model.predict([A,I, ones_full]))
            metric.append("mse")
            value.append(float(MSE))
            iterations.append(i)

            hu = humanity(A, I,n_population, n_household_goods + n_industrial_goods, model, s=True)
            metric.append("$\mathcal{HU}_p$")
            value.append(float(hu))
            iterations.append(i)
            pbar.set_postfix({"hu":"%0.3f" % hu, "mse":"%0.3f" % MSE})
            pbar.update()

            if(i > max_iterations):
                break

    print(len(value), len(metric), len(iterations))
    df = pd.DataFrame({"metric":metric, "value":value, "iteration":iterations})
    df.to_csv("./plot_data/random.csv")
        #x = X_model.predict(np.ones(shape = (1, 3)))
        #hu_1 = calculate_percentages(I - A, x production_df, demand_df, model)

