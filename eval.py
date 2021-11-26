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
        try:
            params[line[0]]=float(line[1])
        except ValueError:
            params[line[0]]=line[1]

path = filename[:-14]

if params['learning_type']=='sarsa':
    Q=np.load(path + "best_quality.npy")
else:
    net=NN()
    net.load_state_dict(torch.load(path + "best_weights.h5"))
    net.eval()
t=0
durations=[]
rewards=[]

episode_duration=params['episode_duration']
learning_step=params['learning_step']
horizon=int(episode_duration/learning_step)
integration_step=params['integration_step']
integration_steps_per_learning_step=int(learning_step/integration_step)
wind_type=params['wind_type']
penalty=params['penalty']
episodes=int(params['episodes'])

n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=pk.n_beta

initial_position=pk.vect(np.pi/6, 0, 50)
initial_velocity=pk.vect(0, 0, 0)
k=pk.kite(initial_position, initial_velocity, wind_type, params)

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
    r.append(initial_position.r)
    theta.append(initial_position.theta)
    phi.append(initial_position.phi)
    alpha.append(S_t[0])
    bank.append(S_t[1])
    beta.append(S_t[2])
    for i in range(horizon):
        if params['learning_type']=='dql':
            tensor_state=torch.tensor(S_t[0:2]).float()
            tensor_state[0]-=(n_attack/2)
            tensor_state[1]-=(n_bank/2)
            tensor_state[0]/=n_attack
            tensor_state[1]/=n_bank
            q=net(tensor_state).reshape(3,3)
            A_t=greedy_action(q.detach().numpy())
        else:
            A_t=greedy_action(Q[S_t])
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
        if i==int(horizon)-1:
            print(ep, "Simulation ended at learning step: ", i, " reward ", cumulative_reward)
            rewards.append(cumulative_reward)
            durations.append(i+1)
        S_t=S_t1

r = np.array(r)
theta = np.array(theta)
phi = np.array(phi)
alpha = np.array(alpha)
bank = np.array(bank)
beta = np.array(beta)
x=np.multiply(r, np.multiply(np.sin(theta), np.cos(phi)))
y=np.multiply(r, np.multiply(np.sin(theta), np.sin(phi)))
z=np.multiply(r, np.cos(theta))

coordinates = np.stack([x,y,z],axis=1)
np.save(path + "eval_traj", coordinates)

controls = np.stack([alpha,bank,beta],axis=1)
np.save(path + "contr_traj", controls)

with open(path+"return_eval.txt", "w") as file:
    for i in range(len(durations)):
        file.write(str(durations[i])+"\t"+str(rewards[i])+"\n")
