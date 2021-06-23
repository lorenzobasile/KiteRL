from learning.deep.models import NN
from utils import dql
import pykite as pk
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys


factors=[1,2,4,8]
runs=3
episodes=1000
gamma=1
eps0=0.01
eta0=0.1
episode_duration=300
learning_step=0.2
horizon=int(episode_duration/learning_step)
integration_step=0.001
integration_steps_per_learning_step=int(learning_step/integration_step)

for f in factors:
    run_params=[]
    while True:
        net=NN(f)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        durations, rewards=dql(net, episodes, horizon, learning_step, integration_step, integration_steps_per_learning_step, pk.vect(np.pi/6, 0, 50), pk.vect(0, 0, 0), pk.vect(10, 0, 0), lr_decay_start=150000, eps_decay_start=200000)
        sys.stdout = save_stdout
        print(f,"  ", durations[-10:])
        if np.mean(np.array(durations[-10:]))>1400:
            count+=1
            for i, param in enumerate(net.parameters()):
                if i==0:
                    param_list=torch.flatten(param).detach().numpy()
            run_params.append(param_list)
            if count==3:
                break
    run_params=np.array(run_params)
    run_params=np.mean(run_params, axis=0)
    plt.figure()
    plt.hist(run_params, bins=np.arange(-20,14)*3)
    plt.show()
    print(run_params.shape)
