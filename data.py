"""
Python module to load the data for training
"""
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from tqdm import tqdm

# Function to load dataset
def load_dataset(csv_path, relative_path):
    """
    Inputs
    ---
    csv_path: path to training data csv
    relative_path: relative path to training data

    Outputs
    ---
    X: Training data numpy array
    y: Training labels numpy array
    """
    # Read CSV lines
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        print("Loading CSV File ...")
        for line in tqdm(reader):
            lines.append(line)
    
    images = []; measurements = []
    print("Loading Data ...")

    # Read from CSV lines
    for line in tqdm(lines):
        # Center Image
        image, measurement = _load_image(line, 0, relative_path)
        images.append(image)
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement_flipped = -1 * measurement
        measurements.append(measurement_flipped)

        # Left Image
        image, measurement = _load_image(line, 1, relative_path)
        images.append(image)
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement_flipped = -1 * measurement
        measurements.append(measurement_flipped)

        # Right Image
        image, measurement = _load_image(line, 2, relative_path)
        images.append(image)
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement_flipped = -1 * measurement
        measurements.append(measurement_flipped)

    X = np.array(images)
    y = np.array(measurements)

    return X, y

# Function to generate a Generator
def load_generator(csv_path, relative_path, batch_size = 5):
    """
    Inputs
    ---
    csv_path: csv file to read data from
    relative_path: relative path of the data
    batch_size: batch size of the generator (factor of 6)

    Outputs
    ---
    generator: generator function
    """
    # Read CSV lines
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        print("Loading CSV File ...")
        for line in tqdm(reader):
            lines.append(line)
    
    train_data, validation_data = train_test_split(lines, test_size=0.2)

    # Define a generator function
    def generator(data, batch_size = batch_size):
        num_data = len(data)
        while True:
            shuffle(data)
            for offset in range(0, num_data, batch_size):
                batch_data = data[offset : offset + batch_size]

                images = []; measurements = []
                # Generate batches
                for batch in batch_data:
                    # Center Image
                    image, measurement = _load_image(batch, 0, relative_path)
                    images.append(image)
                    measurements.append(measurement)

                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)

                    measurement_flipped = -1 * measurement
                    measurements.append(measurement_flipped)

                    # Left Image
                    image, measurement = _load_image(batch, 1, relative_path)
                    images.append(image)
                    measurements.append(measurement)

                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)

                    measurement_flipped = -1 * measurement
                    measurements.append(measurement_flipped)

                    # Right Image
                    image, measurement = _load_image(batch, 2, relative_path)
                    images.append(image)
                    measurements.append(measurement)

                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)

                    measurement_flipped = -1 * measurement
                    measurements.append(measurement_flipped)
                
                X = np.array(images)
                y = np.array(measurements)

                X, y = shuffle(X, y)
                yield (X, y)
    
    return generator(train_data), generator(validation_data), len(train_data), len(validation_data)

# Private function to load image
def _load_image(line, index, relative_path):
    """
    Inputs
    ---
    line: csv line to read data from
    index: decides left, right or center
    relative_path: relative path of the data

    Outputs
    ---
    image: output image
    measurement: output measurement
    """
    source_path = line[index]
    filename = source_path.split('\\')[-1]
    current_path = relative_path + filename
    image = mpimg.imread(current_path)

    if index == 1:
        # Left Image
        correction = 0.2
    elif index == 2:
        # Right Image
        correction = -0.2
    else:
        # Center Image
        correction = 0

    measurement = float(line[3]) + correction

    return image, measurement