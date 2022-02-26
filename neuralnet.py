from torch import nn
from torch.nn import functional as func


class ResidualLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1):
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
        super(RecurrentLayer, self).__init__()
        self.gru = nn.GRU(
            input_size=rnn_size, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(rnn_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.batch_norm(x)
        x = func.relu(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x


class ASRModel(nn.Module):

    def __init__(self, 
                 n_conv_layers, 
                 n_recurrent_layers, 
                 dropout, 
                 hidden_channels, 
                 rnn_dim, 
                 kernel_size, 
                 stride, 
                 conv_padding, 
                 *args,
                 **kwargs
        ):
        super(ASRModel, self).__init__()
        self.conv = nn.Conv2d(2, hidden_channels, kernel_size, stride, conv_padding)
        self.residual_layers = nn.Sequential(*[ResidualLayer(
            in_channels=hidden_channels, out_channels=hidden_channels) for i in range(n_conv_layers)
        ])
        self.conv2 = nn.Conv2d(hidden_channels, 16, kernel_size, stride, conv_padding)
        self.linear = nn.Linear(110, rnn_dim * 2)
        self.recurrent_layers = nn.Sequential(*[RecurrentLayer(rnn_size=rnn_dim * 2, 
            hidden_size=rnn_dim, dropout=dropout, batch_first=(i == 0)) for i in range(n_recurrent_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.Dropout(0.1),
            nn.Linear(rnn_dim, 29),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.residual_layers(x)
        x = self.conv2(x)
        x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.recurrent_layers(x)
        x = self.classifier(x)
        return x