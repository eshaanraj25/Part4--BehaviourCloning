"""
Python file for model definition
"""
from keras.models import Sequential
from keras.layers import (Flatten, Dense, Lambda,
                          Conv2D, MaxPool2D, Cropping2D)

# Function to generate model
def generate_model(input_shape = (160, 320, 3)):
    """
    Inputs
    ---
    input_shape: Shape of the input

    Outputs
    ---
    model: Keras model to train
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = input_shape))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    
    # Generate intermediate model
    model = generate_nvidia_model(model)

    model.add(Dense(1))

    return model

# Function to generate LeNet model
def generate_lenet_model(model):
    """
    Inputs
    ---
    model: model with input defined

    Outputs
    ---
    model: model with LeNet architecture defined
    """
    # Convolution Block 1
    model.add(Conv2D(10, (5, 5), padding = 'valid', activation = 'relu'))    
    model.add(MaxPool2D((2, 2), strides = (2, 2)))      

    # Convolution Block 2
    model.add(Conv2D(32, (5, 5), padding = 'valid', activation = 'relu'))    
    model.add(MaxPool2D((2, 2), strides = (2, 2)))      

    # Flatten
    model.add(Flatten())

    # Dense Layers
    model.add(Dense(32))
    model.add(Dense(8))

    return model

# Function to generate Nvidia model
def generate_nvidia_model(model):
    """
    Inputs
    ---
    model: model with input defined

    Outputs
    ---
    model: model with Nvidia architecture defined

    Reference
    ---
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    # Add convolutional layers
    model.add(Conv2D(24, (5, 5), strides = (2, 2), activation = 'relu')) 
    model.add(Conv2D(36, (5, 5), strides = (2, 2), activation = 'relu'))
    model.add(Conv2D(48, (5, 5), strides = (2, 2), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))

    model.add(Flatten())

    # Add neural network layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))

    return model