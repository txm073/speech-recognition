import json
import os
from tqdm import tqdm
import time

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# Build out the neural network
def build_model(input_shape, output_shape):
    # Model:
    #   Three convolutional layers to extract features from the audio
    #   Batch normalisation to speed up training process and produce more accurate results
    #   Max pooling layer to simplfy/downsize the convolutional output
    #   Flatten layer to transform 3D ouptut from the convs into 1D input for the dense layer
    #   Dense layer with rectified linear activation
    #   30% dropout layer to prevent overfitting - inaccurate predictions
    #   Final dense output layer with softmax classifier to produce a score for each keyword
    model = Sequential()

    # Three convolutional layers

    # Convolutional layer hyper-parameters:
    #   Filters are the number of "pixels" in the output
    #   Kernel size is the size of the window that scans across the input 
    #   Input shape is the number of dimensions and the size of each dimension for the data inputted into the network
    #   Activation "relu" dampens out the extreme values of the convolutional output
    #   Kernel regularizer function helps to prevent overfitting of the data
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    # Downsizes the convolution by factor of 2
    model.add(MaxPool2D((3, 3), (2, 2), padding="same"))

    model.add(Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    # Downsizes the convolution by factor of 2
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    model.add(Conv2D(32, (2, 2), activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    # Downsizes the convolution by factor of 2
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # Flatten output for the dense layer
    model.add(Flatten())
    
    # Units are the number of neurons being inputted into the layer
    model.add(Dense(units=64, activation="relu"))
    
    # 30% of the neurons are disabled to ensure that it isn't just a few neurons doing the classification
    # Makes the network more robust and prevents overfitting
    model.add(Dropout(0.4))

    # Output layer
    # Number of units correspond to the amount of keywords since the output will be a vector of scores/probabilites
    model.add(Dense(units=output_shape, activation="softmax")) 

    # Compile the model
    
    # Optimiser will help the network learn by adjusting the weights and biases between the neurons appropriately
    # Adam optimiser is efficient and the learning rate is the step size for gradient descent
    optimiser = Adam(lr=0.0001)

    # Standard loss function for categorical classification problems
    loss = "sparse_categorical_crossentropy"
    
    # We also want to measure the accuracy of the model on the testing dataset
    # Compile the model with our arguments
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    # Display a summary of the model 
    model.summary()
    
    return model

def split_data(json_path):
    # Load the data from the JSON file
    with open(json_path) as f:
        data = json.load(f)

    # Use the MFCCs as the training values and the labels as the actual values
    # Convert Python lists to Numpy arrays
    x, y = np.array(data["mfccs"]), np.array(data["labels"])

    # Split the x and y data into training data and testing/validation data
    # 10% of the overall dataset is used for testing, the remainder is used for training the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    # Convert 2D arrays into 3D arrays to input into the CNN
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    x_val = x_val[..., np.newaxis]

    return x_train, x_test, x_val, y_train, y_test, y_val

def train(dataset="small", epochs=40, batch_size=32):
    if dataset == "small":
        path = "smaller_dataset.json"
        output_shape = 10
    elif dataset == "large":
        path = "d"
        output_length = 30
    else:
        raise ValueError(
            "Invalid dataset!"
        )

    # Output length is the length of the vector that the network outputs
    # A.K.A the number of different keywords/classes in the dataset
    print("Loading training data...")
    start = time.time()
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(path)
    print("Time taken to load the dataset:", time.time() - start)

    # Build the model
    # Input shape:
    #   X-axis: number of MFCCs
    #   Y-axis: number of audio segments
    #   Z-axis: number of channels (similar to a greyscale image or a mono audio signal)
    input_shape = (x_train.shape[1], x_test.shape[2], 1)
    print(f"Building model with input shape: {input_shape}")
    start = time.time()
    model = build_model(input_shape, output_shape)
    print("Time taken to build the model:", time.time() - start)
    print(f"Input shape: {input_shape}, output shape: {output_shape}")

    # Fit the model on the training data (A.K.A train the model)
    # Epochs indicate the amount of times to loop over the entire dataset
    # Batch size is the amount of inputs the network will receive at once    
    print("Beginning training...")
    start = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    print("Training complete!")
    print(f"Time taken to train the model on {epochs} epoch(s) and a batch size of {batch_size}:", time.time() - start)

    print("Testing the model accuracy and error rate...")
    # Evaluate the model (Measure it's accuracy)
    error, accuracy = model.evaluate(x_test, y_test)
    print(f"Error rate on the test data: {error}, accuracy on the test data: {accuracy}")
    
    # Save the trained model for the future
    model.save("model.h5")

def preprocess(dataset_path):
    # Mappings allow us to index the max value in the network's output vector 
    # To a list of the command words and a number (i) which is assigned to them
    # This allows us to then display the predicted word
    
    # Labels are the correct outputs which the network uses to train on audio files
    
    # MFCCs (Mel-Spectrogram-Cepstral-Coefficients) are arrays of 13 floats which tell us a lot
    # About the timbre of the audio, and they are the numbers that we input 
    # Into the network on the second dimension for training and the third dimension
    # When making predictions
    data = {
        "mappings": [],
        "labels": [],
        "mfccs": []
            }

    # The audio library librosa samples audio at a rate of 22050 samples per second
    # Meaning we are only training on 1-second audio snippets since the network's 
    # Input shape must stay the same and we can enforce the length of the audio signal
    length = 22050

    # Loop over root directories in the dataset
    for i, folder in enumerate(os.listdir(dataset_path)):
        # Skip over the background noise folder
        if folder != "_background_noise_":
            print(f"Processing command: '{folder}'")
            # Add the class name to the mappings list to index later
            data["mappings"].append(folder)
            # Loop through each audio file in the class
            for audio_file in tqdm(os.listdir(os.path.join(dataset_path, folder))):
                audio, sr = librosa.load(os.path.join(dataset_path, folder, audio_file))
                #Enforce the correct length of the audio
                if len(audio) >= length:
                    audio = audio[:length]
                    # Extract 13 MFCCs from the audio signal
                    mfccs = librosa.feature.mfcc(audio, n_mfcc=13, n_fft=2048, hop_length=512)
                    # Transform the array of coefficients from a Numpy array to a Python list
                    coef_list = mfccs.T.tolist()
                    # Append the class label to the JSON list for each file in the class
                    data["labels"].append(i)
                    # Add MFCCs to the JSON data file
                    data["mfccs"].append(coef_list)

    print("Looped over dataset")
    print("Writing to JSON file...")
    # Write data to JSON file
    with open("smaller_data.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    train()