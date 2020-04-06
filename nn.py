from keras import Model
from keras.layers import Dense, Input, Lambda, add, multiply, dot, BatchNormalization, Activation, Layer
from keras.optimizers import SGD, Adam
import keras.backend as K



def activation(x):
    return K.sigmoid(x) * 10



def get_layer(n_inputs):
    visible = Input(shape=(n_inputs,), name = "A")
    eye_visible = Input(shape=(n_inputs,), name = "I")
    v = visible
    e_v = eye_visible

    # v = Lambda(lambda x: x/10)(v)
    # e_v = Lambda(lambda x: x/10)(e_v)
    # v = Dense(10)(v)
    # v = Dense(n_inputs, use_bias = False)(v)


    #v = LeakyReLU()(visible)
    ones = Input(shape=(3,), name = "Dummy")

    X = Dense(1, use_bias=False)(ones)
    X = Dense(n_inputs, use_bias=False, name = "x")(X)
    v = add([e_v, Lambda(lambda x: -x)(v)])



    out = dot([X, v], axes = -1)
    #out = BatchNormalization()(out)

    return out, visible, eye_visible,  ones, X

def nn(n_inputs):
    #n_inputs = 100000

    out, visible, eye_visible, ones,  X = get_layer(n_inputs)
    #out, visible, eye_visible = get_layer(n_inputs, out)

    model = Model(inputs=[visible, eye_visible,  ones], outputs=[out], name="model")
    model.compile(optimizer=Adam(0.001), loss="mse")

    X_model = Model(inputs=[ones], outputs=[X], name="X_model")

    #model.compile(optimizer=SGD(lr=0.1), loss="mse")
    print(model.summary())
    #exit()
    return model, X_model