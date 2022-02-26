import torch
from torch import nn
from torch.nn import functional as func


class ResidualLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super(ResidualLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.expanded_channels = self.out_channels * self.expansion
        self.downsampling = downsampling
        self.identity = nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut() else None
        
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut():
            residual = self.shortcut(x)
        x = self.identity(x)
        x += residual
        x = func.relu(x)
        return x


class RecurrentLayer(nn.Module):

    def __init__(self, rnn_size, hidden_size, dropout, batch_first):
        super(ReccurentLayer, self).__init__()
        self.gru = nn.GRU(
            input_size=rnn_size, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = func.gelu(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x


"""
class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_size, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_size)
        self.birnn_layers = nn.Sequential(*[
            ReccurentLayer(rnn_size=rnn_size if i==0 else rnn_size*2,
                             hidden_size=rnn_size, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_size*2, rnn_size),  # birnn returns rnn_size*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_size, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x
"""


hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_size": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 5
}

conv = nn.Conv2d(2, 32, 3, 3, 1)