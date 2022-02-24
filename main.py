import os, sys
import warnings
import random
import numpy as np
from tqdm import tqdm

import torch 
from torch import nn, optim
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta

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
        channels = audio.shape[0]
        new_signal = torchaudio.transforms.Resample(old_sr, new_sr)(audio[:1, :])
        if channels == 2:
            second_channel = torchaudio.transforms.Resample(old_sr, new_sr)(audio[1:, :])
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
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(audio)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spec_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for i in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for i in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec


class TextProcess:
    
    @classmethod
    def create_charmaps(cls):
        chars = "' abcdefghijklmnopqrstuvwxyz"
        cls.int2char = {i: c for i, c in enumerate(chars)}
        cls.char2int = {c: i for i, c in enumerate(chars)}

    @classmethod
    def text2int(cls, text):
        return [cls.char2int[char] for char in text.lower().strip()]

    @classmethod
    def int2text(cls, arr):
        return "".join([cls.int2char[i] for i in arr])


TextProcess.create_charmaps()


class SpeechDataset(ta.datasets.LIBRISPEECH):

    def __init__(self, *args, **kwargs):
        super(SpeechDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, i):
        audio, sr, label, *_ = super(SpeechDataset, self).__getitem__(i)
        return process(audio, sr), label  

    def process(self, audio, sr, augment=True):
        audio = AudioProcess.resample(audio, sr, SR)
        audio = AudioProcess.to_stereo(audio)
        audio = AudioProcess.resize(audio, SR, DURATION)
        return audio, sr


SR = 16000
DURATION = 12000
CHANNELS = 2
TRAIN_SET = SpeechDataset("train-set", "train-clean-100", download=True)
TEST_SET = SpeechDataset("test-set", "test-clean", download=True)
print(TRAIN_SET[0])