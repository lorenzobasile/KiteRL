import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, scale=1, init_q=None):
        super().__init__()
        self.layer1=nn.Linear(2,9*scale)
        #self.layer2=nn.Linear(8*scale,16)
        #self.layer3=nn.Linear(16,9)
        if init_q is not None:
            nn.init.uniform_(self.layer3.weight, a=0, b=0)
            nn.init.uniform_(self.layer3.bias, a=init_q, b=init_q)
        self.activation=nn.LeakyReLU()

    def forward(self, x):
        #x=self.activation(self.layer1(x))
        #x=self.activation(self.layer2(x))
        return self.layer1(x)
