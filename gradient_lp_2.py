import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras import Model
from keras.layers import Dense, Input, Lambda, add, multiply
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from keras.constraints import Constraint
import keras.backend as K
import tensorflow as tf
from utils import full_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class MaxAvailable(Constraint):
    def __init__(self, max_value):
        self.max_value = max_value
    def __call__(self, w):
        desired = (K.clip(w, 0, (self.max_value)))
        return desired



def get_layer(n_inputs, current_available):
    visible = Input(shape=(n_inputs,))

    X = Dense(1, use_bias=True)
    out = X(visible)

    return out, visible, X

def nn(n_inputs, current_available):
    #n_inputs = 100000

    out, visible, X = get_layer(n_inputs, current_available)
    #out, visible, eye_visible = get_layer(n_inputs, out)

    model = Model(inputs=[visible], outputs=[out], name="model")
    model.compile(optimizer="Adam", loss="mse")
    #print(model.summary())
    #exit()
    return model, X


# def nn(n_inputs, current_available):
#     visible = Input(shape=(n_inputs,))
#     eye_visible = Input(shape=(n_inputs,))
#     ones_visible = Input(shape=(n_inputs,))
#     W = Dense(1, use_bias=False)
#     X = Dense(n_inputs, use_bias=False, trainable=False)
#
#
#     out = multiply([X(ones_visible),W(visible)])
#     I = W(eye_visible)
#
#     out = add([I, Lambda(lambda x: -x)(out)])
#
#     model = Model(inputs=[visible, eye_visible, ones_visible], outputs=[out], name="model")
#     model.compile(optimizer=SGD(lr=1), loss="mse")
#
#     return model, X


if __name__ == "__main__":
    production_df = pd.read_csv("data/butter_production.csv")
    demand_df = pd.read_csv("data/butter_demand.csv")

    overproduction = 2
    matrix = full_matrix(production_df.copy(), demand_df, overproduction)

    A = matrix[:, :-1]
    I = np.eye(A.shape[0])
    y = matrix[:, -1]
    ones_full = np.ones(shape=(A.shape[0], 3))

    model, w = nn(len(A.T), np.array([[120,140]]).T)
    X = I-A
    c = StandardScaler()
    c.fit(X)
    X = c.transform(X)

    #print(c.data_range_)
    #y = c.transform(np.array([y])).T
    #print(y.shape)




    model.fit(X, y, epochs=10000, batch_size=50, shuffle=True)

    print(mean_squared_error(y, model.predict(X)))
    #print()
    #print()
    k = w.get_weights()
    #k[0][1] = 10
    print(k)
    #print(w.set_weights(k))
    #print("actual output", model.predict([X, eyes]))
