import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras import Model
from keras.layers import Dense, Input, Lambda, add, multiply, dot, BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error
from keras.constraints import Constraint
import keras.backend as K
import tensorflow as tf




def get_layer(n_inputs, current_available):
    visible = Input(shape=(n_inputs,))



    ones = Input(shape=(3,))

    X = Dense(1, use_bias=False)(ones)
    #X = BatchNormalization()(X)
    X = Dense(n_inputs, use_bias=False)(X)
    #X = BatchNormalization()(X)

    out = dot([X, visible], axes = -1)



    #out = add([I, Lambda(lambda x: -x)(out)])
    return out, visible, ones, X

def nn(n_inputs, current_available):
    #n_inputs = 100000

    out, visible, ones,  X = get_layer(n_inputs, current_available)
    #out, visible, eye_visible = get_layer(n_inputs, out)

    model = Model(inputs=[visible, ones], outputs=[out], name="model")
    model.compile(optimizer=Adam(0.001), loss="mse")

    X_model = Model(inputs=[ones], outputs=[X], name="X_model")

    print(model.summary())
    #exit()
    return model, X_model




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
    y = matrix[:, -1]
    ones = np.ones(shape = (A.shape[0],3) )
    one = np.ones(shape = (1, 3))
    X = I - A
    print(ones.shape)
    #exit()

    model, X_model = nn(len(X.T), np.array([[500,500]]).T)
    model.fit([X, ones], y, epochs=10000, batch_size=100, shuffle=True, verbose=False)

    print(mean_squared_error(y, model.predict([X, ones])))

    x = X_model.predict(one)
    print(x)
    #print()
    #k = w.get_weights()
    #k[0][1] = 10
    #print(k)
    #print(w.set_weights(k))
    print("actual output", model.predict([X, ones]))
