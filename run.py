import numpy as np
from learning.algorithms import *
from learning.models import NN
from argparse import ArgumentParser
import sys
import os
import matplotlib.pyplot as plt
from test import test 


def main(args):
    
    np.random.seed(0)
    path=args.path
    if not os.path.exists(path):
        os.makedirs(path)
    n_attack = pk.coefficients.shape[0]
    n_bank = pk.bank_angles.shape[0]
    n_beta = pk.n_beta
    window_size = 30
    initial_position = pk.vect(np.pi/6, 0, 20)
    initial_velocity = pk.vect(0, 0, 0)
    
    
    if args.alg == 'sarsa':
        k = pk.kite(initial_position, initial_velocity, args.wind,continuous=False)
        Q = np.ones((n_attack, n_bank, n_beta, 3, 3))*0
        Q_traj, Q, durations, rewards = sarsa(k, Q, args, initial_position, initial_velocity)
        np.save(path + "best_quality", Q)
        np.save(path + "quality_traj", Q_traj)
        test(k, args)
        
    elif args.alg == 'td3': 
        k = pk.kite(initial_position, initial_velocity, args.wind,continuous=True)
        durations, rewards = td3(k, args, initial_position, initial_velocity) 
        test(k, args)
        
    else:
        k = pk.kite(initial_position, initial_velocity, args.wind,continuous=True)
        net = NN()
        net, durations, rewards = dql(k, net, args, initial_position, initial_velocity)
        torch.save(net.state_dict(), path + "best_weights.h5")

    plt.figure(figsize=(10,6))
    plt.title("Cumulative reward")
    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Energy (kWh)', fontsize=16)
    plt.plot(rewards, 'o')

    smooth = np.convolve(rewards, np.ones(100), "valid")/100
    plt.plot(smooth, color='red', lw=1)
    plt.savefig(args.path+'return.png', dpi=200)
    plt.show()

    
    plt.figure(figsize=(10,6))
    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Duration (s)', fontsize=16)
    plt.plot(durations, 'o')
    smooth = np.convolve(durations, np.ones(100), "valid")/100
    plt.plot(smooth, color='red', lw=1)
    plt.savefig(args.path+'durations.png', dpi=200)
    plt.show()


    with open(path + "return.txt", "w") as file:
        for i in range(len(durations)):
            file.write(str(durations[i]) + "\t" + str(rewards[i]) + "\n")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="./results/const/")
    parser.add_argument("--alg", default="sarsa")
    parser.add_argument("--wind", default="const") #const, lin or turbo
    parser.add_argument('--step',type=float,default=0.1)
    parser.add_argument('--critic_lr',type = float, default = 0.001) 
    parser.add_argument('--actor_lr',type = float, default = 0.001) 
    parser.add_argument("--episodes", type=int, default=1e4)
    parser.add_argument("--eval_episodes", type=int, default=1e1)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--lrstart", type=int, default=1000)
    parser.add_argument("--epsstart", type=int, default=500000)
    parser.add_argument("--lrrate", type=float, default=0.8)
    parser.add_argument("--epsrate", type=bool, default=0.8)
    parser.add_argument("--personalizedlr", type=bool, default=True)
    parser.add_argument('--range_actions',action='append',default = None) 
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    

