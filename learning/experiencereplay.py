import torch
from time import time

class ExperienceBuffer:
    def __init__(self, size, ns, na):
        self.data=torch.zeros((size, ns+na+1+ns+1))
        self.size=size
        self.current_size=0
        self.insertion_point=0
    def insert(self, observation):
        self.data[self.insertion_point]=observation
        if self.current_size<self.size:
            self.current_size+=1
        self.insertion_point+=1
        self.insertion_point=self.insertion_point%self.size
    def get_batch(self, batch_size):
        if batch_size>self.size:
            return self.data
        indices = torch.randperm(self.size)[:batch_size]
        return self.data[indices]
