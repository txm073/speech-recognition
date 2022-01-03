# Script to process batches of raw audio data and prepare it for input into the neural network

import librosa
import numpy as np
from torch import nn
import torchaudio
import tqdm

import os
import sys
import time


# Convert strings of text into integer arrays
# And re-convert output probablities into characters
# By assigning an integer to each character in the alphabet 
chars = "' abcdefghijklmnopqrstuvwxyz"
int2char = dict((i, c) for i, c in enumerate(chars))
char2int = dict((c, i) for i, c in enumerate(chars))

def text2ints(text):
    return [char2int[char] for char in text if char in chars]

def ints2text(array):
    return "".join([int2char[i] for i in array])


# Convert the incoming audio feed into mel-frequency spectrograms
# This is a pictorial representation of sound 
# Which can be inputted into the CNN to extract a useful feature map
# This will hopefully cause the RNN to make better predictions
class MelSpecAugment(nn.Module):
    
    def __init__(self):
        # Initialise parent class
        super().__init__()

        # Convert data to mel spectrograms
        self.spec = torchaudio.transforms.MelSpectrogram()

        # Augment data by cutting out small portions
        # Along the time and frequency domains
        # This will make the model more generalisable
        # As it forces the model to make predictions with imperfect data
        self.spec_aug = nn.Sequential(
                    # Trial and error for hyper-params
                    torchaudio.transforms.TimeMasking(10), 
                    torchaudio.transforms.FrequencyMasking(10)
                        ) 

        self.spec_aug2 = nn.Sequential(
                    torchaudio.transforms.TimeMasking(10), 
                    torchaudio.transforms.FrequencyMasking(10),
                    torchaudio.transforms.TimeMasking(10), 
                    torchaudio.transforms.FrequencyMasking(10)
                        )

    # Method called automatically
    # When data is passed through the initialised class
    def forward(self, x):
        x = self.spec(x)
        ## Add small value to prevent x = infinity
        #x = np.log(x + 1e-14)
        
        # Augment data using time and frequncy masking
        # Choose random amount of augmentation
        # Whether the process is repeated twice or not
        if random.random() > 0.5:
            return self.spec_aug(x)
        return self.spec_aug2(x)


# Read MP3 data as numpy array 
# Ready for it to be converted to a mel-spectrogram
def read_mp3(file):
    audio, sr = librosa.load(file)
    return audio

# Main
os.chdir(os.path.dirname(__file__))
if __name__ == "__main__":
    array = read_mp3("testfile.wav")
    print("Read MP3 file as array")
    spec = MelSpecAugment()
    array = spec.forward(array)
