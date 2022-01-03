import torch
from torch import nn
from torch.nn import functional as F

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(128, 256, num_layers=1)
        self.fc2 = nn.Linear(256, 29)

    def forward(self, x):
        x = self.cnn(x)
        print(x.shape)
        return
        x = self.fc1(x)
        x = self.lstm(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    net = Network()
    x = torch.randn([1, 128, 251, 1])

    x = net(x)
        