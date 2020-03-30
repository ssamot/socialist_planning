import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras import Model
from keras.layers import Dense, Input, Lambda, add, multiply
from keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error
from keras.constraints import Constraint
import keras.backend as K
import tensorflow as tf
from sklearn import preprocessing

class MaxAvailable(Constraint):
    def __init__(self, max_value):
        self.max_value = max_value
    def __call__(self, w):
        desired = (K.clip(w, 0, (self.max_value)))
        return desired



def get_layer(n_inputs, current_available):
    visible = Input(shape=(n_inputs,))
    #eye_visible = Input(shape=(n_inputs,))
    X = Dense(1, use_bias=False, kernel_constraint=MaxAvailable(current_available))
    out = X(visible)
    #I = X(eye_visible)

    #out = add([I, Lambda(lambda x: -x)(out)])
    return out, visible, X

def nn(n_inputs, current_available):
    #n_inputs = 100000

    out, visible, X = get_layer(n_inputs, current_available)
    #out, visible, eye_visible = get_layer(n_inputs, out)

    model = Model(inputs=[visible], outputs=[out], name="model")
    model.compile(optimizer=Adam(0.001), loss="mse")
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
    #dep_df = pd.read_csv("data/dummy_flows.csv")
    dep_df = pd.read_csv("data/butter2.csv")



    matrix = dep_df.values[:, 1:]


    A = matrix[:, :-1]
    I = np.eye(A.shape[0])
    y = matrix[:, -1:]
    X = I - A


    model, w = nn(len(X.T), np.array([[1000]*len(A.T)]).T)
    model.fit([X], y, epochs=1000, batch_size=200, shuffle=True, verbose=False)

    print(mean_squared_error(y, model.predict([X])))
    #print()
    #print()
    k = w.get_weights()[0]
    #k[0][1] = 10
    print(k)
    #print(y_scaler.inverse_transform(k))
    #print(w.set_weights(k))
    print("actual output", model.predict([X]))
