import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import numpy as np
from keras import Model
from keras.layers import Dense, Input, LSTM as NormalLSTM, RepeatVector, TimeDistributed, CuDNNLSTM, Lambda, Reshape, \
    concatenate, BatchNormalization, \
    GaussianNoise, Dropout, Flatten, Activation, Bidirectional, CuDNNGRU, GRU, Conv1D, add
import tensorflow as tf
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
import keras.backend as K
from sklearn.metrics import mean_squared_error


#
# class Leo(Dense):
#     def call(self, inputs):
#         #output = K.dot(K.eye(size = 2) - inputs, self.kernel)
#         output = K.dot(K.eye(size=2) , self.kernel) - K.dot(inputs, self.kernel)
#         if self.use_bias:
#             output = K.bias_add(output, self.bias, data_format='channels_last')
#         if self.activation is not None:
#             output = self.activation(output)
#         return output

def nn(n_inputs):
    visible = Input(shape=(n_inputs,))
    eye_visible = Input(shape=(n_inputs,))
    X = Dense(1, use_bias=False)
    out = X(visible)
    I = X(eye_visible)

    out = add([I, Lambda(lambda x: -x)(out)])
   
    model = Model(inputs=[visible, eye_visible], outputs=[out], name="model")
    model.compile(optimizer=SGD(lr = 1), loss="mse")

    return model, X


if __name__ == "__main__":
    dep_df = pd.read_csv("data/io.csv")

    matrix = dep_df.values[:, 1:]
    X = matrix[:, :-1]
    eyes = np.eye(X.shape[0])
    # print(eyes)
    # exit()
    y = matrix[:, -1]
    #y = y/10.0
    # print(X)
    # print(y)
    # exit()
    model, w = nn(len(X.T))
    # print(model2.predict(X))
    # print(y)
    # exit()
    model.fit([X, eyes], y, epochs=100, batch_size=1000, shuffle=True)

    print(mean_squared_error(y, model.predict([X, eyes])))
    # print(model.predict([X, eyes]), y)
    print(w.get_weights())
