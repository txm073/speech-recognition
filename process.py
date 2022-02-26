import os, sys
import warnings
import random
import numpy as np
from tqdm import tqdm

import torch 
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
        return *self.process(audio, sr), label  

    def process(self, audio, sr, augment=True):
        audio = AudioProcess.resample(audio, sr, SR)
        audio = AudioProcess.to_stereo(audio)
        audio = AudioProcess.resize(audio, SR, DURATION)
        if augment:
            audio = AudioProcess.time_shift(audio, 0.2)
        spec = AudioProcess.spectrogram(audio, sr)
        if augment:
            spec = AudioProcess.specaugment(spec)
        return spec, sr


SR = 16000
DURATION = 14000
CHANNELS = 2
EPOCHS = 5
BATCH_SIZE = 16
TRAIN_SET = SpeechDataset("train-set", "train-clean-100", download=True)
TEST_SET = SpeechDataset("test-set", "test-clean", download=True)
TRAIN_LOADER = DataLoader(TRAIN_SET, batch_size=BATCH_SIZE, shuffle=True)
TEST_LOADER = DataLoader(TEST_SET, batch_size=BATCH_SIZE, shuffle=True)

from model import ResidualLayer, conv

res = ResidualLayer(32, 32)
for i, (batch, sr, labels) in enumerate(TRAIN_LOADER):
    output = res(torch.randn(1, 32, 4, 23))
    print(output)
    break