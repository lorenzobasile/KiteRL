import numpy as np
from learning.algorithms import *
from learning.models import NN
import sys

filename = sys.argv[1]

with open(filename) as file:
    params = {}
    lines = file.readlines()
    for line in lines:
        line = line.split()
        print(line)
        try:
            params[line[0]] = float(line[1])
        except ValueError:
            params[line[0]] = line[1]

path = filename[:-14]
n_attack = pk.coefficients.shape[0]
n_bank = pk.bank_angles.shape[0]
n_beta = pk.n_beta
max_power = params['max_power']
window_size = 30
initial_position = pk.vect(np.pi / 6, 0, 50)
initial_velocity = pk.vect(0, 0, 0)
wind_type = params['wind_type']
k = pk.kite(initial_position, initial_velocity, wind_type, params)
if params['learning_type'] == 'sarsa':
    Q = np.ones((n_attack, n_bank, n_beta, 3, 3))
    Q *= max_power
    Q_traj, Q, durations, rewards = sarsa(k, Q, params, initial_position, initial_velocity)
    np.save(path + "best_quality", Q)
else:
    net = NN()
    net, durations, rewards, Q_traj, L = dql(k, net, params, initial_position, initial_velocity)
    torch.save(net.state_dict(), path + "best_weights.h5")
np.save(path + "quality_traj", Q_traj)
np.save(path+"loss_traj", L)


with open(path + "return.txt", "w") as file:
    for i in range(len(durations)):
        file.write(str(durations[i]) + "\t" + str(rewards[i]) + "\n")
