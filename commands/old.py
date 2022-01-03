import string
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import librosa, librosa.display
import os

charset = "' " + string.ascii_lowercase
int2char = dict((i, c) for i, c in enumerate(charset))
char2int = dict((c, i) for i, c in enumerate(charset))


def text2int(text):
    return [char2int[char] for char in text.lower()]


def int2text(int_list):
    return "".join([int2char[i] for i in int_list])


def augmentspec(spec, time=2, freq=2, show=False):
    freq_mask = tfio.audio.freq_mask(spec, param=freq)
    time_mask = tfio.audio.time_mask(freq_mask, param=time)
    if show:
        plt.figure()
        plt.imshow(time_mask.numpy())
        plt.show()

    return time_mask.numpy()

def melspec(file):
    audio, sr = librosa.load(file)
    mel_spec = librosa.feature.melspectrogram(audio)
    mel_spec = librosa.power_to_db(mel_spec)
    librosa.display.specshow(mel_spec)
    plt.show()


# Using tf
def _melspec(file, show=False):
    audio = tfio.audio.AudioIOTensor(file)

    audio_slice = audio[100:]

    # Remove last dimension
    audio_tensor = tf.squeeze(audio_slice, axis=[-1])

    tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
 
    position = tfio.audio.trim(
        tensor, 
        axis=0, 
        epsilon=0.1 # Uppermost value considered noise 
            )

    start = position[0]
    stop = position[1]

    processed = tensor[start:stop]

    fade = tfio.audio.fade(
        processed, fade_in=1000, fade_out=2000, mode="logarithmic")

    # Convert to spectrogram
    spectrogram = tfio.audio.spectrogram(
        fade, 
        nfft=512, # Number of fast-fourier-transforms (extracting individual frequencies) 
        window=512, # Number of audio segments
        stride=256 # Distance between the segments
            )

    # Convert to mel-spectrogram
    # Mel-spectrograms use the mel scale
    # Perceptually relevant logarithmic scale for pitch
    # E.G. 500Hz to 100Hz equal to 10000Hz to 10500Hz 
    mel_spec = tfio.audio.melscale(
        spectrogram,  
        rate=16000, # Sample rate of audio
        mels=100, # Number of mels on the scale
        fmin=0, # Minimum frequency in Hz 
        fmax=8000 # Maximum frequency in Hz
            )

    # Convert to decibel scale mel-spectrogram
    dbscale_spec = tfio.audio.dbscale(
        mel_spec, top_db=80)

    if show:
        plt.figure()
        plt.imshow(dbscale_spec.numpy())
        plt.show()

    return dbscale_spec

os.chdir(os.path.dirname(__file__))
augmentspec(_melspec("testfile.wav"), time=15, freq=15, show=True)
