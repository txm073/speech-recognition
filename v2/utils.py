# Script for preprocesing data before it is fed into the accoustic model
# Also used for post-processing data after it is outputted from the language model
# Includes utility methods
# TODO import nltk
# Rules-based NLP to improve transcriptions

import torch
from torch import nn
import torchaudio
import numpy as np


class TextProcess:
    # Convert training labels into arrays to input into the network
    # And to decode the network's inputted character probabilities into text
    
    def __init__(self):
        # Character map
        self.chars = "' abcdefghijklmnopqrstuvwxyz"
        # Index to character map
        self.int2char = {i: c for i, c in enumerate(self.chars)}
        # Inverse - character to index map
        self.char2int = {c: i for i, c in enumerate(self.chars)}

    def int_to_text(self, ints):
        # Convert an integer array to a text sequence based on the index to character map
        return "".join([self.int2char[i] for i in ints])

    def text_to_ints(self, text):
        # Convert a text sequence into an integer array based on the character to index map
        return [self.char2int[char] for char in text.lower()]

    def onehotencode(self, text):
        # One-Hot encodes a text sequence
        # Uses an array of zeros for each character, 
        # And then a one for that character's index in the character map
        # Example:
        # "abc" = [[1, 0, 0 ... 0, 0, 0], [0, 1, 0 ... 0, 0, 0], [0, 0, 1 ... 0, 0, 0]]
        output = []
        for char in text:
            row = [0] * len(self.chars)
            row[self.chars.index(char)] = 1
            output.append(row)
        return output

    def onehotdecode(self, array):
        # Decodes an array of One-Hot encoded vectors
        text = ""
        for vec in array:
            text += self.chars[vec.index(1)]
        return text


train_audio_transforms = nn.Sequential(
    # Transform the training data into Mel-Frequency Spectrograms
    # These can act as a pictorial representation of sound, and they can be fed into a CNN
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    # We also need to use Spec-Augment, a data augmentation technique for better generalisation
    # This involves removing parts of the data across the time and frequency domain
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
    # Destroying sections of the data will increase the size of the dataset
    # It will also force the network to make better predictions with imperfect data
)

# Do not use Spec-Augment if the data is for validation, just convert to melspec
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_process = TextProcess()


def preprocess(data, data_type="train"):
    # Function for processing each batch of data before it is fed through the model
    spectrograms, labels, input_lengths, label_lengths = [], [], [], []
    # Unpack each object which is yielded by the data loader
    for audio, sr, text, *_ in data:
        # Convert data to spectrograms
        if data_type == "train":
            # If training then also augment the spectrogram
            spec = train_audio_transforms(audio).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(audio).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        # Encode the label
        label = torch.Tensor(text_process.text_to_ints(text.lower()))
        labels.append(label)
        # Lengths are important as they need to be provided to the CTC loss function
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))
    # Pad the spectrograms and swap around the last 2 dimensions
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def _lev(token1, token2):  
    # Function to calculate Levenshtein distance 
    # An algorithm used to display the difference/distance between two sequences
    # Displays the number of changes (substitutions, insertions or deletions)
    # That are needed for the two sequences to exactly match 
    m, n = len(token1), len(token2)  
    # Initialise distances matrix as a 2D NumPy array
    # Array dimensions: m x n where m and n are the lengths of the words/tokens
    matrix = np.zeros([m+1, n+1])
    for i in range(m):
        matrix[i][0] = i
    for i in range(n):
        matrix[0][i] = i
    # First row and column are the respective token's enumerations/count
    a, b, c = 0, 0, 0
    # Iterate over the matrix excluding the element at 0, 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            # If element to the left and above are equal
            # Update the current element to also be equal
            if token1[i-1] == token2[j-1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                # Element to the left
                a = matrix[i][j - 1]
                # Element directly above
                b = matrix[i - 1][j]
                # Element diagonally above and left
                c = matrix[i - 1][j - 1]
                # Update the current position according to the algorithm
                if a <= b and a <= c:
                    matrix[i][j] = a + 1
                elif b <= a and b <= c:
                    matrix[i][j] = b + 1
                else:
                    matrix[i][j] = c + 1
    # Overall number of changes needed
    # Equal to bottom-right (m, n) element of the distance matrix
    return matrix[m][n] 

def word_errors(reference, pred, ignore_case=False, delimiter=" "):
    # Calculate the Levenshtein Distance 
    # Between reference sequence and predicted sequence in word-level
    if ignore_case:
        reference = reference.lower()
        pred = pred.lower()

    ref_words = reference.split(delimiter)
    pred_words = pred.split(delimiter)

    return _lev(ref_words, pred_words), len(ref_words)

def char_errors(reference, pred, ignore_case=False, remove_space=False):
    # Calculate the Levenshtein Distance 
    # Between reference sequence and predicted sequence in char-level
    if ignore_case:
        reference = reference.lower()
        pred = pred.lower()

    join_char = " "
    if remove_space == True:
        join_char = ""

    reference = join_char.join(filter(None, reference.split(" ")))
    pred = join_char.join(filter(None, pred.split(" ")))

    return _lev(reference, pred), len(reference)

def wer(reference, pred, ignore_case=False, delimiter=" "):
    # Word Error Rate (WER) is a metric used alongside the CTC Loss function
    # Defined as the edit distance (Levenshtein Distance) / length of reference sequence
    # Distances and lengths calculated at word level
    edit_distance, ref_length = word_errors(reference, pred, ignore_case, delimiter)
    try:
        return edit_distance / ref_length
    except ZeroDivisionError:
        raise ValueError(
            "Length of reference sequence cannot be 0"
        ) from None

def cer(reference, pred, ignore_case=False, remove_space=False):
    # Character Error Rate (CER) is a metric used alongside the CTC Loss function
    # Defined as the edit distance (Levenshtein Distance) / length of reference sequence
    # Distances and lengths calculated at character level
    edit_distance, ref_length = char_errors(reference, pred, ignore_case, remove_space)
    try:
        return edit_distance / ref_length
    except ZeroDivisionError:
        raise ValueError(
            "Length of reference sequence cannot be 0"
        ) from None

