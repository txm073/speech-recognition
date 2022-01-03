import os
import sys

import librosa
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_io as tfio
import pickle


chars = " 'abcdefghijklmnopqrstuvwxyz"
int2char = {i: c for i, c in enumerate(chars)}
char2int = {c: i for i, c in enumerate(chars)}

def load_tsv(path):
    df = pd.read_table(path)
    return [(row["path"], row["sentence"]) for _, row in df.iterrows()]

def process_text(text):
    return np.array([char2int[char] for char in text if char in chars])

def one_hot_encode(y):
    output = []
    if type(y) is str:
        for char in y:
            char_prob = np.zeros(29)
            char_prob[char2int[char]] = 1
            output.append(char_prob)

    else:
        for i in y:
            char_prob = np.zeros(29)
            char_prob[i] = 1
            output.append(char_prob)
    return np.array(output)

def process(x, y, sample_rate=8000, duration=8, onehot=False):
    target_length = int(duration * sample_rate)

    # Load the audio file as a Numpy array
    audio_folder = "C:/Users/User/Desktop/Common Voice/clips"
    audio, _ = librosa.load(f"{audio_folder}/{x}.mp3", sr=sample_rate)
    audio_length = len(audio)
    if audio_length > target_length:
        return None
    else:
        pad = target_length - audio_length

    audio = np.concatenate((audio, np.zeros(pad)))

    mel_spec = librosa.feature.melspectrogram(audio,
        sr=sample_rate, n_fft=2048, hop_length=256
    )
    log_mel_spec = np.log(mel_spec + 1e-10)

    time_mask = tfio.audio.time_mask(log_mel_spec, param=8)
    freq_mask = tfio.audio.freq_mask(time_mask, param=8).numpy()

    if onehot is True:
        proc_label = onehot_encode(y)
    else:
        proc_label = process_text(y)

    folder = "C:/Users/User/Desktop/Common Voice/arrays/"
    np.savez_compressed(folder + x + ".npy", a=np.array([freq_mask, proc_label], dtype=object))


if __name__ == "__main__":
    path = "C:/Users/User/Desktop/Common Voice/validated.tsv"
    data = load_tsv(path)
    print("Loaded data!")
    for i, (path, label) in tqdm(enumerate(data)):    
        try:
            process(path, label)
        except:
            with open("index.txt", "w") as f:
                f.write(str(i))
