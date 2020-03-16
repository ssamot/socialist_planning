import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras import Model
from keras.layers import Dense, Input, Lambda, add
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error


def nn(n_inputs):
    visible = Input(shape=(n_inputs,))
    eye_visible = Input(shape=(n_inputs,))
    X = Dense(1, use_bias=False)
    out = X(visible)
    I = X(eye_visible)

    out = add([I, Lambda(lambda x: -x)(out)])

    model = Model(inputs=[visible, eye_visible], outputs=[out], name="model")
    model.compile(optimizer=SGD(lr=1), loss="mse")

    return model, X


if __name__ == "__main__":
    dep_df = pd.read_csv("data/io.csv")

    matrix = dep_df.values[:, 1:]
    X = matrix[:, :-1]
    eyes = np.eye(X.shape[0])
    y = matrix[:, -1]
    model, w = nn(len(X.T))
    model.fit([X, eyes], y, epochs=100, batch_size=1000, shuffle=True)

    print(mean_squared_error(y, model.predict([X, eyes])))
    print(w.get_weights())
