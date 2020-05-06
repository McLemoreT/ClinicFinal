"""
Spyder Editor

This is a temporary script file.
"""
from keras.datasets import fashion_mnist

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys
import os.path
from pathlib import Path
import datetime

def init (thickness, size, epochs, batches, iteration):
    
    print('start')
 
    model_file = Path(path + str(iteration) + 'model.h5')
    if model_file.exists():
         model = load_model(model_file)
         print(model_file)
         print('Loaded old model')
    else:
         
         print('Model Not found')
         print('Creating Model')
         CreateModel(thickness, size, epochs, batches, iteration)

def CreateModel (thickness, size, epochs, batches, iteration):
    #Set variables




    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

	#Build the model
    model=Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation=None, input_shape=(28,28,1), strides=1))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', strides=1))
    model.add(Conv2D(filters=32, kernel_size=7, padding='same', activation=None, strides=1))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax')) #Softmax should always be last!

    print(model.summary())
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Save stats
    X_valid, y_valid = x_train[:batches], y_train[:batches]
    X_train2, y_train2 = x_train[batches:], y_train[batches:]
	
    history = model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batches, epochs=epochs)
    
    #evaluation
    
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])
    print('Saving weights')
    model.save_weights(path + str(iteration) + "modelweights.h5")
    print('model weights saved')
    print('Saving entire model')
    model.save(path + str(iteration) + 'model.h5')
    f = open(path +  str(iteration) + "Model Stats.txt", "a")
    f.write(dt_date + "\n")
    f.write("batch size: " + str(batches) + "\n")
    f.write("epoch: " + str(epochs) + "\n")
    f.write("Optimizer: SGD" + "\n") #Automate this
    #model.summary(print_fn=myprint)
    
   # Open the file
    with open(path +  str(iteration) + "Model Stats.txt",'a') as fh:
       # Pass the file handle in as a lambda function to make it callable
       model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    
    f.write(str(scores[1]))
    f.write("\n" + "\n")
    #f.write(str(model.summary()))
    f.close()
    print('Model saved')
    
    #history = model.fit(X_train2, y_train2,, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    fig1.savefig(path + str(iteration) + "accuracy.png", bbox_inches='tight', dpi=300, quality = 100)
    plt.show()
    
    # summarize history for loss
    fig2 = plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss?', 'val_loss?'], loc='upper left')
    fig2.savefig(path + str(iteration) + "loss.png", bbox_inches='tight', dpi=300, quality = 100)
    plt.show()
#This stuff should be global
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = ROOT_DIR + "\\"

dt_date = datetime.datetime.now()
dt_date = dt_date.strftime('%d/%m/%y %I:%M %S %p')