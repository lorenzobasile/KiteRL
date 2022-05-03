import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(3,256, bias=True)
        self.layer2=nn.Linear(256,32, bias=True)
        self.layer3=nn.Linear(32,9, bias=False)
        self.activation=nn.ReLU()

    def forward(self, x):
        x=self.activation(self.layer1(x))
        x=self.activation(self.layer2(x))
        return self.layer3(x)
