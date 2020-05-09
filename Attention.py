

from keras.applications import VGG16
import keras
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from keras.models import load_model
# Build the VGG16 network with ImageNet weights
model = load_model('5model.h5')
print('Model loaded.')
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
y_test = keras.utils.to_categorical(y_test, 10) #10 for number of classes
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test = x_test/255
import numpy as np
titles = ["T-shirt or top","Trouser", "Pullover", 
          "Dress", "Coat", "Sandal", "Shirt", "Sneaker", 
          "Bag", "Ankle boot"]

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

from matplotlib import pyplot as plt

class_idx = 0
indices = np.where(y_test[:,class_idx] == 1.)[0]

# pick some random input from here.
idx = indices[0]

# Lets sanity check the picked image.
from matplotlib import pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

plt.imshow(x_test[idx][..., 0])


from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'visualized_layer')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_test[idx])
# Plot with 'jet' colormap to visualize as a heatmap.
plt.imshow(grads, cmap='jet')


for class_idx in np.arange(10):    
    indices = np.where(y_test[:, class_idx] == 1.)[0]
    idx = indices[0]

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(x_test[idx][..., 0])

    i=0
    ax[i].set_title(titles[class_idx])
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, 
                               seed_input=x_test[idx])
    
    modifier = 'vanilla'
    ax[i+1].set_title(modifier)    
    ax[i+1].imshow(grads, cmap='jet')

