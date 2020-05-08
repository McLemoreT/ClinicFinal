# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:05:14 2020

@author: Tyler
"""


from keras.applications import VGG16

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from keras.models import load_model
# Build the VGG16 network with ImageNet weights
model = load_model('5model.h5')
print('Model loaded.')
import numpy as np


from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

from matplotlib import pyplot as plt

titles = ["T-shirt or top","Trouser", "Pullover", 
          "Dress", "Coat", "Sandal", "Shirt", "Sneaker", 
          "Bag", "Ankle boot"]
%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'visualized_layer')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 1
#img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
#plt.imshow(img[..., 0])



#img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.))
#plt.imshow(img[..., 0])

#img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
 #                          tv_weight=1., lp_norm_weight=0., verbose=True)
#plt.imshow(img[..., 0])

for output_idx in np.arange(10):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.))
    plt.figure()
    plt.title('Networks perception of {}'.format(titles[output_idx]))
    plt.imshow(img[..., 0])
    plt.savefig(titles[output_idx]+"loss.png", bbox_inches='tight', dpi=300, quality = 100)