"""
@author: Tanja Stanic
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import load_model
import help_functions as hf

def make_model():
    cnn_model = Sequential()
    
    # dodavanje prvoh ulaznog sloja
    # nas ulazni sloj je slika 28x28
    cnn_model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
    cnn_model.add(Conv2D(28, (3, 3), activation='sigmoid'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
   
    # kreiranje srednjeg sloja sa 56 neurona koji ima relu aktivacionu funkcuju
    cnn_model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    cnn_model.add(Conv2D(56, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
  
    cnn_model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    cnn_model.add(Conv2D(56, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
  
    # kreiranje skrivenog sloja sa 128 neurona koji ima relu aktivacionu funkcuju
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(0.5))
    
    # postoji deset izlaza 
    # cifre : [0,1,2,3,4,5,6,7,8,9]
    # poslednji sloj ima softmax aktivacionu funkciju
    cnn_model.add(Dense(10, activation='softmax'))
  
    return cnn_model

# treniranje neuronske mreze
def neural_network():
    # koristi se mnist set podataka
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train = hf.image_scale(X_train)
    X_test = hf.image_scale(X_test)
    
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)


    # pravljenje modela
    cnn_model = make_model()
    # ovucavanje modela
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    cnn_model.fit(X_train, Y_train, batch_size=256, epochs=10, verbose=1, shuffle=False) 
    score = cnn_model.evaluate(X_test, Y_test, verbose=0)          
    print(score)
  
    cnn_model.save_weights("model.h5")

    return 1,cnn_model
    
def load_neural_network(model_name):
    loaded_model = load_model(model_name)
    return loaded_model
    