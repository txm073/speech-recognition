import os, sys
import warnings
import random
import numpy as np
from tqdm import tqdm

import torch 
from torch import nn, optim, autograd
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta

from neuralnet import ASRModel

warnings.simplefilter("ignore")

try:
    os.chdir(os.path.dirname(sys.argv[0]))
    os.mkdir("test-set")
    os.mkdir("train-set")
except OSError:
    pass


class AudioProcess:

    @staticmethod
    def to_mono(audio):
        return audio[:1, :] if (audio.shape[0] == 2) else audio

    @staticmethod
    def to_stereo(audio):
        return torch.cat((audio, audio)) if (audio.shape[0] == 1) else audio

    @staticmethod
    def resample(audio, old_sr, new_sr):
        if old_sr == new_sr:
            return audio
        channels = audio.shape[0]
        new_signal = ta.transforms.Resample(old_sr, new_sr)(audio[:1, :])
        if channels == 2:
            second_channel = ta.transforms.Resample(old_sr, new_sr)(audio[1:, :])
            new_signal = torch.cat((new_signal, second_channel))
        return new_signal

    @staticmethod
    def resize(audio, sr, ms):
        channels, audio_length = audio.shape
        max_len = sr // 1000 * ms

        if (audio_length > max_len):
            audio = audio[:, :max_len]

        elif (audio_length < max_len):
            pad_begin_len = random.randint(0, max_len - audio_length)
            pad_end_len = max_len - audio_length - pad_begin_len

            pad_begin = torch.zeros((channels, pad_begin_len))
            pad_end = torch.zeros((channels, pad_end_len))

            audio = torch.cat((pad_begin, audio, pad_end), 1)
        
        return audio

    @staticmethod
    def time_shift(audio, max_shift):
        _, audio_length = audio.shape
        shift_amt = int(random.random() * max_shift * audio_length)
        return audio.roll(shift_amt)

    @staticmethod
    def spectrogram(audio, sr, n_mels=64, n_fft=1024, hop_len=None):
        top_db = 80
        spec = ta.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(audio)
        spec = ta.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def specaugment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for i in range(n_freq_masks):
            aug_spec = ta.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for i in range(n_time_masks):
            aug_spec = ta.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec


class TextProcess:
    
    @classmethod
    def create_charmaps(cls):
        cls.chars = "' abcdefghijklmnopqrstuvwxyz0"
        cls.int2char = {i: c for i, c in enumerate(cls.chars, 1)}
        cls.char2int = {c: i for i, c in enumerate(cls.chars, 1)}

    @classmethod
    def text2int(cls, text):
        return [cls.char2int[char] for char in text.lower()]

    @classmethod
    def int2text(cls, arr):
        return "".join([cls.int2char[i] for i in arr])

    @classmethod
    def resize(cls, arr, length):
        if len(arr) > length:
            return arr[:length]
        else:
            for i in range(length - len(arr)):
                arr.append(hparams["padding_char"])
            return arr

    @classmethod
    def unpad(cls, arr):
        return [n.item() for n in arr if n != hparams["padding_char"]]


TextProcess.create_charmaps()


class SpeechDataset(ta.datasets.LIBRISPEECH):

    def __init__(self, *args, **kwargs):
        super(SpeechDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, i):
        item = super(SpeechDataset, self).__getitem__(i)
        audio, sr, text, *_ = item
        audio = AudioProcess.to_stereo(audio)
        audio = AudioProcess.resample(audio, sr, hparams["sample_rate"])
        audio = AudioProcess.resize(audio, hparams["sample_rate"], hparams["audio_duration"])
        spec = AudioProcess.spectrogram(audio, sr)
        spec = AudioProcess.specaugment(spec)
        label = TextProcess.text2int(text)
        label = TextProcess.resize(label, hparams["label_length"])
        return spec, torch.Tensor(label)


class CTCDecoder:
    pass


hparams = {
    "epochs": 5, # Loop over the dataset
    "batch_size": 16, # Size of batches of data inputted
    "audio_duration": 14000, # Duration of the processed audio in milliseconds
    "sample_rate": 16000, # Sample rate of the processed audio data
    "padding_char": 29, # Special character for padding labels
    "blank_char": 0, # Special character to denote silence in the audio
    "label_length": 256, # Length of labels in characters
    "n_conv_layers": 3, # Number of residual convolutional layers
    "n_recurrent_layers": 3, # Number of bidirectional GRU layers
    "dropout": 0.1, # Proportion of neurons to randomly disable each pass
    "hidden_channels": 32, # Number of filters in the hidden layer of the conv layers
    "rnn_dim": 256, # Predict 1 character for each timestep
    "kernel_size": 3, # Size of the window performing convolutions
    "stride": 2, # Stride of the convolutional window
    "conv_padding": 1, # Padding around the kernel
    "output_classes": len(TextProcess.chars), # Number of output classes for the model
    "learning_rate": 0.001 # Rate of descent when updating weights
}

train_set = SpeechDataset("train-set", "train-clean-100", download=True)
test_set = SpeechDataset("test-set", "test-clean", download=True)
train_loader = DataLoader(train_set, batch_size=hparams["batch_size"], shuffle=True)
test_loader = DataLoader(test_set, batch_size=hparams["batch_size"], shuffle=True)

model = ASRModel(**hparams)
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=hparams["learning_rate"])

for i, (batch, labels) in enumerate((train_loader)):
    outputs = model(batch)
    ctc_data = (
        outputs, 
        labels,  
        torch.Tensor([len(output) for output in outputs]).to(torch.int32), 
        torch.Tensor([len(TextProcess.unpad(label)) for label in labels]).to(torch.int32)
    )
    print(outputs[0][255].shape)
    break