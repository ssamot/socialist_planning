import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras import Model
from keras.layers import Dense, Input, Lambda, add, multiply, dot, BatchNormalization, Activation, Layer
from keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error
from keras.constraints import Constraint
import keras.backend as K
import tensorflow as tf

class LeakyReLU(Layer):

    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape


def get_layer(n_inputs, current_available):
    visible = Input(shape=(n_inputs,))
    v = visible
    #v = LeakyReLU()(visible)
    ones = Input(shape=(3,))

    X = Dense(1, use_bias=False)(ones)
    X = Dense(n_inputs, use_bias=False)(X)
    #X = LeakyReLU(alpha=0.0001)(X)
    #X = Activation("relu")(X)
    X = Activation("sigmoid")(X)

    out = dot([X, v], axes = -1)
   # out = Activation("relu")(out)

    return out, visible, ones, X

def nn(n_inputs, current_available):
    #n_inputs = 100000

    out, visible, ones,  X = get_layer(n_inputs, current_available)
    #out, visible, eye_visible = get_layer(n_inputs, out)

    model = Model(inputs=[visible, ones], outputs=[out], name="model")
    model.compile(optimizer=Adam(0.01), loss="mse")

    X_model = Model(inputs=[ones], outputs=[X], name="X_model")

    #model.compile(optimizer=SGD(lr=0.1), loss="mse")
    print(model.summary())
    #exit()
    return model, X_model

def full_matrix(production_df, demand_df, overcompletion):
    industries = list(production_df["Type"])
    indices = list(demand_df["Name"])

    print(production_df)
    print(demand_df)

    for idx in indices:
        production_df.insert(production_df.shape[1] - 1, idx, 0)

    for idx in indices:
        row = [0]*len(production_df.columns)
        #print(row)
        row[0] = idx
        row[-1] = overcompletion
        #print(row)
        k = dict(zip(production_df.columns, row))
        #print(k)

        production_df = production_df.append(k, ignore_index=True)

    basic_industries = list(demand_df.columns)[1:]
    print(basic_industries)

    for industry in basic_industries:
        for name in indices:
            value = float(demand_df.loc[demand_df["Name"] == name, industry])
            production_df.loc[production_df["Type"] == industry, name] = value



    print(production_df.to_latex())
    matrix = production_df.values[:, 1:]
    return matrix

if __name__ == "__main__":
    production_df = pd.read_csv("data/butter_production.csv")
    demand_df = pd.read_csv("data/butter_demand.csv")

    matrix = full_matrix(production_df, demand_df, 1.0)


    A = matrix[:, :-1]
    I = np.eye(A.shape[0])
    y = matrix[:, -1]
    ones = np.ones(shape = (A.shape[0],3) )
    one = np.ones(shape = (1, 3))
    X = I - A


    model, X_model = nn(len(X.T), np.array([[500,500]]).T)
    model.fit([X, ones], y, epochs=1000, batch_size=30, shuffle=True, verbose=False)

    print("MSE", mean_squared_error(y, model.predict([X, ones])))

    x = X_model.predict(one)
    l = dict(zip(production_df.columns[1:-(len(demand_df.columns) )], x.T))

    solution = pd.DataFrame(l)
    print(solution)
    #print("A",x.dot(A))
    print("num", X[0], x[0])
    X[0][-2] = 0
    print("num", X[0].dot(x[0]))

    print("actual output", model.predict([X, ones]))
