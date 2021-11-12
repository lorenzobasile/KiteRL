import numpy as np
import pykite as pk
from utils import *
import matplotlib.pyplot as plt
import os
import sys

filename=sys.argv[1]

with open(filename) as file:
    params={}
    lines=file.readlines()
    for line in lines:
        line=line.split()
        print(line)
        try:
            params[line[0]]=float(line[1])
        except ValueError:
            params[line[0]]=line[1]


np.random.seed(0)

path="./plots/test/"
n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=pk.n_beta
max_power=params['max_power']
eta0=params['eta0']
gamma=params['gamma']
eps0=params['eps0']
episode_duration=params['episode_duration']
learning_step=params['learning_step']
horizon=int(episode_duration/learning_step)
integration_step=params['integration_step']
integration_steps_per_learning_step=int(learning_step/integration_step)
wind_type=params['wind_type']
window_size=30
Q=np.ones((n_attack, n_bank, n_beta, 3, 3))
Q*=(max_power*2)

durations=[]
rewards=[]
episodes=int(params['episodes'])
t=0
theta0=[]
phi0=[]
r0=[]
for j in range(episodes):
    cumulative_reward=0
    initial_position=pk.vect(np.pi/6, 0, 50)
    initial_velocity=pk.vect(0, 0, 0)
    k=pk.kite(initial_position, initial_velocity, wind_type)
    initial_beta=k.beta()
    S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), initial_beta)
    k.C_l, k.C_d = pk.coefficients[S_t[0],0], pk.coefficients[S_t[0],1]
    k.psi = np.deg2rad(pk.bank_angles[S_t[1]])
    A_t=eps_greedy_policy(Q, S_t, eps0)
    for i in range(horizon):
        t+=1
        eps=scheduling(eps0, t, params['eps_decay_start'], exp=params['eps_decay_rate'])
        eta=scheduling(eta0, t, params['eta_decay_start'], exp=params['eta_decay_rate'])
        if j==episodes-1:
            theta0.append(k.position.theta)
            phi0.append(k.position.phi)
            r0.append(k.position.r)
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        if not sim_status==0:
            R_t1 = scheduling(-3000000, i, horizon/4)
            cumulative_reward+=R_t1
            print(j, "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
            rewards.append(cumulative_reward)
            durations.append(i)
            Q=terminal_step(Q, S_t, A_t, R_t1, eta)
            break
        S_t1 = (new_attack_angle, new_bank_angle, k.beta())
        R_t1 = k.reward(learning_step)
        cumulative_reward+=R_t1
        A_t1=eps_greedy_policy(Q, S_t1, eps)
        if i==int(horizon)-1:
            Q=terminal_step(Q, S_t, A_t, R_t1, eta)
            print(j, "Simulation ended at learning step: ", i, " reward ", cumulative_reward)
            rewards.append(cumulative_reward)
            durations.append(i)
        else:
            Q=step(Q, S_t, A_t, R_t1, S_t1, A_t1, eta, gamma)
        S_t=S_t1
        A_t=A_t1
print(np.max(Q[3,6,0]))

try:
    os.mkdir(path)
except OSError:
    pass
theta0=np.array(theta0)
phi0=np.array(phi0)
r0=np.array(r0)

x=np.multiply(r0, np.multiply(np.sin(theta0), np.cos(phi0)))
y=np.multiply(r0, np.multiply(np.sin(theta0), np.sin(phi0)))
z=np.multiply(r0, np.cos(theta0))

plt.figure()
plt.plot(x)
plt.show()
plt.figure()
plt.plot(y)
plt.show()
plt.figure()
plt.plot(z)
plt.show()

plot_trajectory(theta0[::10], phi0[::10], r0[::10], save=path+"traj.png")

plt.figure()
plt.plot(durations, 'o')
plt.plot(np.convolve(durations, np.ones(window_size), 'valid') / window_size)
plt.savefig(path+"durations.png")
plt.show()
plt.figure()
plt.plot(rewards, 'o')
plt.plot(np.convolve(rewards, np.ones(window_size), 'valid') / window_size)
plt.savefig(path+"rewards.png")
plt.show()
