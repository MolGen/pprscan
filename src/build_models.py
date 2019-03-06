from keras.layers import (Activation, BatchNormalization, Conv1D, Dense,
                          Dropout, Flatten, GlobalMaxPool1D, MaxPooling1D)
from keras.models import Sequential
from keras.optimizers import Adam, SGD, Adagrad
from keras.regularizers import l2


def build_model_conv1D(): #cnn2
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=7, strides=1, padding="same", input_shape=(35, 21)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(5))

    model.add(Flatten())
    
    model.add(Dense(256, use_bias=False, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128, use_bias=False, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))

    model.add(Dense(12, kernel_initializer="he_uniform"))
    model.add(Activation('softmax'))

    #adagrad = Adagrad(lr=0.0025)
    adagrad = Adagrad(lr=0.01)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adagrad,
                  metrics=['accuracy'])
    
    return model

def build_model_conv1D(input_shape=(35, 21), output_shape=12):
    model = Sequential(name='conv1D')

    model.add(Conv1D(filters=64, kernel_size=7, strides=1, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(5))

    #model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding="same"))
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    #model.add(Dropout(0.1))
    #model.add(MaxPooling1D(4))

    model.add(Flatten())

    model.add(Dense(256, use_bias=False))#, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, use_bias=False))#, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation="softmax")) ### sigmoid # y.shape[1]
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    
    model.compile(optimizer=sgd, #'adam',
                  loss='categorical_crossentropy', ### binary_crossentropy
                  metrics=['accuracy'])

    return model

def build_model_mlp1():
    model = Sequential(name='mlp')
    model.add(Dense(256, use_bias=False, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01), input_shape=(735,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, use_bias=False, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128, use_bias=False, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(12, activation="softmax"))   
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', ### binary_crossentropy
                  metrics=['accuracy'])

    return model

def build_model_mlp2():
    model = Sequential()

    model.add(Dense(256, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01), input_shape=(735,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128, use_bias=False, kernel_initializer="he_uniform", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.2)) #

    model.add(Dense(12))
    model.add(Activation('softmax'))
    
    adagrad = Adagrad(lr=0.0025)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adagrad,
                  metrics=['accuracy'])

    return model