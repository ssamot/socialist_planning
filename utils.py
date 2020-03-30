import numpy as np



def flow(X, y, batch_size=128, balance = True):
    k = 0
    while True:
        X = X[k:k+batch_size]
        y = y[k:k+batch_size]

        # dense matrices
        X = np.array(X)
        y = np.array(y)
        #print(y)
        yield X, y