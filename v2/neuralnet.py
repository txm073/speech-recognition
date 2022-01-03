# ResNet + Recurrent neural network model 
from torch import nn 
from torch.nn import functional as fn


class Model(nn.Module):
    # ASR accoustic neural network model
    # Architecture:
    #   - Input: Spectrogram of shape 
    #   - Residual convolutional layers
    #   - Linear (fully connected) layers
    #   - Bidirectional GRU layers
    #   - Linear layers + dropout layers
    #   - Softmax classifier output per time step

    def __init__(self, 
                n_cnn_layers, 
                n_rnn_layers, 
                rnn_dim, 
                n_class, 
                n_feats, 
                stride, 
                dropout 
    ):
        super().__init__()
        n_feats //= 2
        # Initial CNN layer
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  
        # Residual CNN layers
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        # Linear layers
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        # Bidirectional GRU layers
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=(i==0))
            for i in range(n_rnn_layers)
        ])
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  
        # Swap second and third dimensions (features, time)
        x = x.transpose(1, 2) 
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


class CNNLayerNorm(nn.Module):
    # Normalisation layer to normalise output of the 2D Conv layer
    # Improves training time and the network's generalisation

    def __init__(self, n_feats):
        super().__init__()
        # `n_feats` indicates the number of output neurons after the LayerNorm
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        x = x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 
        return x


class ResidualCNN(nn.Module):
    # Custom 'residual' CNN layer has benefits over regular CNN
  
    def __init__(self, 
                 in_channels, # Number of input dimensions
                 out_channels, # Number of output dimensions
                 kernel, # Convolutional window size
                 stride, # Gap between window steps
                 dropout, # Percentage of neurons to deactivate
                 n_feats # Shape of LayerNorm output
    ):
        super().__init__()

        # Convolutional layers - feature extraction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        # Dropout layers - disables certain neurons to stop overfitting
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Layer normalisation - smoother gradients, more generalisation
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)
    

    def forward(self, x):
        # Residual architecture includes concatenating the inputs to the outputs
        # Helps with vanishing gradients, performs better in large networks
        residual = x  
        x = fn.gelu(self.layer_norm1(x))
        x = self.dropout1(x)
        x = self.conv1(x)
        x = fn.gelu(self.layer_norm2(x))
        x = self.dropout2(x)
        x = self.conv2(x)
        x += residual
        return x                                                                                            


class BidirectionalGRU(nn.Module):
    # RNN layer to output probabilites over a series of time steps
    # Gated Recurrent Units to help with short term memory in RNNs
    # Similar to LSTM but less intensive, but with comparable results
    # Bidirectional to allow the layer to use context from before and after

    def __init__(self, 
                 rnn_dim, # Number of neurons in the RNN input layer
                 hidden_size, # Number of neurons in the hidden layers  
                 dropout, # Percentage of neurons to deactivate
                 batch_first # Whether to output with the batch as dim 1
    ):
        super().__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = fn.gelu(self.layer_norm(x))
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x
