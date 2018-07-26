import keras.backend as K

def swish(x):
    return x*K.sigmoid(x)