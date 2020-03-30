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

class MaxAvailable(Constraint):
    def __init__(self, max_value):
        self.max_value = max_value
    def __call__(self, w):
        desired = (K.clip(w, 0, (self.max_value)))
        return desired



def get_layer(n_inputs, current_available):
    visible = Input(shape=(n_inputs,))
    eye_visible = Input(shape=(n_inputs,))
    X = Dense(1, use_bias=False, kernel_constraint=MaxAvailable(current_available))
    out = X(visible)
    I = X(eye_visible)

    out = add([I, Lambda(lambda x: -x)(out)])
    return out, visible, eye_visible, X

def nn(n_inputs, current_available):
    #n_inputs = 100000

    out, visible, eye_visible, X = get_layer(n_inputs, current_available)
    #out, visible, eye_visible = get_layer(n_inputs, out)

    model = Model(inputs=[visible, eye_visible], outputs=[out], name="model")
    model.compile(optimizer=SGD(lr=0.01), loss="mse")
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
    #dep_df = pd.read_csv("data/io.csv")
    dep_df = pd.read_csv("data/butter.csv")

    matrix = dep_df.values[:, 1:]
    X = matrix[:, :-1]
    eyes = np.eye(X.shape[0])
    y = matrix[:, -1]

    model, w = nn(len(X.T), np.array([[120,140]]).T)
    model.fit([X, eyes], y, epochs=10000, batch_size=1, shuffle=True)

    print(mean_squared_error(y, model.predict([X, eyes])))
    #print()
    #print()
    k = w.get_weights()
    #k[0][1] = 10
    print(k)
    #print(w.set_weights(k))
    print("actual output", model.predict([X, eyes]))
