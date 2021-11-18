import numpy as np
import pykite as pk
from utils import *
import matplotlib.pyplot as plt
import os
from models import NN
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

path = filename[:-14]

if params['learning_type']=='sarsa':
    Q=np.load(path + "best_quality.npy")
else:
    net=NN()
    net.load_state_dict(path + "best_weights.h5")

t=0
durations=[]
rewards=[]

eps0=params['eps0']

episode_duration=params['episode_duration']
learning_step=params['learning_step']
horizon=int(episode_duration/learning_step)
integration_step=params['integration_step']
integration_steps_per_learning_step=int(learning_step/integration_step)
wind_type=params['wind_type']
penalty=params['penalty']
episodes=int(params['episodes'])

r = []
theta = []
phi = []
alpha = []
bank = []
beta = []

ep=0
while ep<=episodes:

    cumulative_reward=0
    ep+=1
    k.reset(initial_position, initial_velocity, wind_type, params)
    initial_beta=k.beta()
    S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), initial_beta)
    k.C_l, k.C_d = pk.coefficients[S_t[0],0], pk.coefficients[S_t[0],1]
    k.psi = np.deg2rad(pk.bank_angles[S_t[1]])
    A_t=eps_greedy_policy(Q, S_t, eps0)

    r.append(initial_pos[2])
    theta.append(initial_pos[0])
    phi.append(initial_pos[1])
    alpha.append(S_t[0])
    bank.append(S_t[1])
    beta.append(S_t[2])

    for i in range(horizon):
        t+=1
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        r.append(k.position.r)
        theta.append(k.position.theta)
        phi.append(k.position.phi)
        S_t1 = (new_attack_angle, new_bank_angle, k.beta())
        alpha.append(S_t1[0])
        bank.append(S_t1[1])
        beta.append(S_t1[2])
        if not sim_status==0:
            R_t1 = scheduling(-penalty, i, horizon/4)
            cumulative_reward+=R_t1
            print(ep, "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
            rewards.append(cumulative_reward)
            durations.append(i+1)
            break
        R_t1 = k.reward(learning_step)
        cumulative_reward+=R_t1
        A_t1=eps_greedy_policy(Q, S_t1, eps)
        if i==int(horizon)-1:
            print(ep, "Simulation ended at learning step: ", i, " reward ", cumulative_reward)
            rewards.append(cumulative_reward)
            durations.append(i+1)
        S_t=S_t1
        A_t=A_t1

r = np.array(r)
theta = np.array(theta)
phi = np.array(phi)
alpha = np.array(alpha)
bank = np.array(bank)
beta = np.array(beta)
x=np.multiply(r, np.multiply(np.sin(theta), np.cos(phi)))
y=np.multiply(r, np.multiply(np.sin(theta), np.sin(phi)))
z=np.multiply(r, np.cos(theta))

coordinates = np.vstack(x,y,z)
print(coordinates[0])
np.save(coordinates, "eval_traj")

controls = np.vstack(alpha,bank,beta)
np.save(controls, "contr_traj")
