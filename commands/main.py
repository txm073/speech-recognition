import librosa
import scipy.io.wavfile as wav
import sounddevice as sd
import os

import numpy as np
from tensorflow.keras.models import load_model

SAMPLE_RATE = 22050

def predict_keyword(filename=None):
    keywords = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]    
    if filename is None:
        print("Recording...")
        audio = sd.rec(frames=SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype=np.float64)
        sd.wait()
    else:
        audio, _ = librosa.load(filename)

    pad = np.zeros(int(SAMPLE_RATE - len(audio))) 
    audio = np.concatenate([audio, pad])

    audio = audio.reshape(SAMPLE_RATE,)

    input_data = librosa.feature.mfcc(audio, 
                                      n_mfcc=13, 
                                      hop_length=512, 
                                      n_fft=2048).reshape([1, 44, 13, 1])

    output = model.predict(input_data)
    print(keywords[np.argmax(output)])


def record(filename=None):
    print("Recording...")
    # Since the network requires 1 second audio clips as input 
    # The number of samples is equal to the number of frames
    # Also the CNN expects the third dimension (number of audio channels) to be 1
    rec = sd.rec(frames=SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    if filename is not None:
        wav.write(filename, SAMPLE_RATE, rec)

if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(__file__))
    except OSError:
        pass
    model = load_model("model.h5")
    input("Model loaded")
    predict_keyword("yes-5.wav")
    #predict_keyword(r"C:\Users\Tom\Desktop\Common\Datasets\up\00b01445_nohash_0.wav")