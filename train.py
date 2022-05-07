"""
Main python file
"""
from math import ceil
from model import generate_model
from data import load_dataset, load_generator
import matplotlib.pyplot as plt

# Load the data
# X, y = load_dataset("data/driving_log.csv", "data/IMG/")

# Load the data generator
BATCH_SIZE = 10
train_generator, validation_generator, train_length, validation_length = load_generator("data/driving_log.csv", "data/IMG/", BATCH_SIZE)

# Get the model
model = generate_model()
model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(X, y, validation_split = 0.2, shuffle = True, epochs = 4)
# Fit generator
history_object = model.fit_generator(
    train_generator, steps_per_epoch = ceil(train_length / BATCH_SIZE),
    validation_data = validation_generator, validation_steps = ceil(validation_length / BATCH_SIZE),
    epochs = 5, verbose = 1
)

# Save the model
model.save('model.h5')

# Plot loss functions
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('visualization.png')
plt.show()