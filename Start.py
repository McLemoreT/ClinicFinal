"""
Created on Wed Mar 25 15:54:24 2020

@author: Tyler
"""

print('begin')
import datetime
import os
import sys
from pathlib import Path
import tensorflow as tf
#import Trainer
import Network
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

thickness = 1   #NEtwork thickness, unused
size = 12
epochs = 100  #How many times/epochs?
batches = 64   #Batch size for batch processing 
iteration = 4 #How many times has the program run in a loop?




#Do we have cuda?
print(tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#Start
Network.init(thickness, size, epochs, batches, iteration)

from keras.models import load_model

#Test an input image
model = load_model('1model.h5')

test_image_generator = ImageDataGenerator(
            rescale=1./255) # Generator for our tes data
            
test_generator = test_image_generator.flow_from_directory(
        directory = "./tests",
        target_size=(28, 28),
        #color_mode="grayscale",
        color_mode="grayscale", 
        batch_size=1,
        class_mode="categorical",
        shuffle=False
)
legend = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
out = model.predict_generator(test_generator)
files = test_generator.filepaths
for j in range ( np.size(files) ):
    maxi = 0
    for k in range ( np.size(out[j]) ):
        if out[j][k] > out[j][maxi]:
            maxi = k
    print(files[j], end = ", ")
    print(maxi, end = ", ")
    print(legend[maxi])
    
#print(out)
#print(files)

if not os.path.exists('MNIST/images'):
   os.makedirs('MNIST/images/')
os.chdir('MNIST/images/')




    