import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, scale=1, init_q=None):
        super().__init__()
        self.layer1=nn.Linear(105,32, bias=True)
        self.layer2=nn.Linear(32,9, bias=True)
        #self.layer4=nn.Linear(64,32, bias=True)
        #self.layer3=nn.Linear(32,9, bias=True)
        #self.layer2=nn.Linear(16*scale,9, bias=False)
        #self.layer3=nn.Linear(16,9)
        if init_q is not None:
            nn.init.uniform_(self.layer1.weight, a=0, b=0)
            nn.init.uniform_(self.layer1.bias, a=init_q, b=init_q)
        self.activation=nn.ReLU()

    def forward(self, x):
        x=self.activation(self.layer1(x))
        #x=self.activation(self.layer2(x))
        #x=self.activation(self.layer4(x))
        return self.layer2(x)
