import os
import tempfile
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import activations
from keras.models import load_model
from tensorflow.python.framework import ops

def normalize(array):
    """
    Returns the original passed in array with its values normalized within the
    range of [0,1]
    
    args:
        array: The numpy array to normalize
    """
    arr_min = np.min(array)
    arr_max = np.max(array)
    return (array - arr_min) / (arr_max - arr_min + K.epsilon())

def linearize_activation(model, custom_objects=None):
    """
    Returns the passed in model with its final layer actvation set to linear.
    This helps to achieve clearer results
    
    args:
        model: The model to modify
        custom_objects: Any custom objects utilized by the model
    """
    model.layers[-1].activation = activations.linear
    try:
        model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)

def compute_gradient(model, output_index, input_image):
    """
    Computes and returns the gradients of the input image from a single
    back-propogation pass through the model, with respect to the output index
    
    args:
        model: The model to perform back-propogation on
        output_index: The class which to obtain gradients from
        input_image: The input which the gradients will come from
    """
    # Grab input and outputs
    input_tensor = model.input
    output_tensor = model.output
    wrt_tensor = K.identity(input_tensor)

    # Define loss
    loss_fn = K.mean(output_tensor[:, output_index])

    # Compute gradient
    grad_fn = K.gradients(loss_fn, input_tensor)[0]

    # Normalize gradients
    grad_fn = K.l2_normalize(grad_fn)

    # Define the function to compute the gradients
    compute_fn = K.function([input_tensor],
                            [loss_fn, grad_fn, wrt_tensor])

    # Perform gradient descent
    computed_values = compute_fn([input_image])
    loss, grads, wrt_value = computed_values
    print(grads)

    # "Deprocess input"
    return grads

def visualize_saliency(model, output_index, input_image, custom_objects=None):
    """
    An implementation of Deep Inside Convolutional Networks: Visualising Image
    Classification Models and Saliency Maps. Through a single back-propagation
    pass of a given Keras model, this library visualizes the spatial support of
    a given class in a given image.

    args:
        model: The model to visualize saliency for
        output_index: The class which to visualize saliency with respect to
        input_image: The input to visualize spatial support on
        custom_objects: Custom objects utilized by the model
    """
    model = linearize_activation(model, custom_objects)
    if len(input_image.shape) != len((1, ) + K.int_shape(self.input_tensor)[1:]):
        input_image = np.expand_dims(input_image, 0)
    grads = compute_gradient(model, output_index, input_image)
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    grads = np.max(grads, axis=channel_idx)
    
    # Invert the gradients. The greater influence a pixel has on the output, the smaller the
    # gradient will be. Since we want to visualize areas with the greatest influence, 
    # inverting the gradients will provide a larger gradient for these smaller values.
    grads = 1.0 - grads
    return normalize(grads)[0]
