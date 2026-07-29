import torch
import torch.nn as nn


class CustomCNNDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64 * 32 * 32 , 16) 
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(16,5)

    def forward(self, x):
     x = self.features(x)
     x = self.flatten(x)
     x = self.dense1(x)
     x = self.relu(x)
     x = self.dense2(x)   # lowercase x now
     return x