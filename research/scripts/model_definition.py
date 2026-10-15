"""Protocol architecture. No training or inference occurs on import."""
import torch
from torch import nn


class ECGClassifier(nn.Module):
    def __init__(self):
        super().__init__();layers=[]
        for a,b in zip([1,16,32,64,64],[16,32,64,64,64]):
            layers.extend([nn.Conv1d(a,b,7,stride=2,padding=3),nn.BatchNorm1d(b),nn.ReLU()])
        self.features=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool1d(1);self.dropout=nn.Dropout(.2);self.head=nn.Linear(64,1)

    def forward(self,x):
        return self.head(self.dropout(self.pool(self.features(x)).squeeze(-1))).squeeze(-1)
