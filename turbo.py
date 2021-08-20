import pykite as pk
from utils import dql, dql_eval, plot_trajectory
from learning.deep.models import NN, NN5
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=pk.n_beta
gamma=1
eps0=0.01
eta0=0.00001
episode_duration=300
learning_step=0.2
horizon=int(episode_duration/learning_step)
integration_step=0.001
integration_steps_per_learning_step=int(learning_step/integration_step)

torch.manual_seed(10)
np.random.seed(10)
net=NN()
durations, rewards,_,_,_=dql(net, 'sgd', 2000, horizon, learning_step, integration_step, integration_steps_per_learning_step, pk.vect(np.pi/6, 0, 50), pk.vect(0, 0, 0), 10, lr_decay_start=4000000, eps_decay_start=4000000, eps=eps0, lr=eta0)

plt.figure()
plt.title("Duration")
s_durations=np.array(durations)*0.2
plt.plot(s_durations, 'o')
plt.plot(np.convolve(s_durations, np.ones(300), 'valid') / 300)
plt.xlabel("episode")
plt.ylabel("s")
plt.savefig('duration.png', dpi=100)
plt.figure()
kwh_rewards=np.array(rewards)/3.6e6
plt.title("Cumulative reward")
plt.plot(kwh_rewards, 'o')
plt.plot(np.convolve(kwh_rewards, np.ones(300), 'valid') / 300)
plt.xlabel("episode")
plt.ylabel("kWh")
plt.savefig('reward.png', dpi=100)
