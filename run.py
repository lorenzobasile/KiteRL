import numpy as np
from learning.algorithms import *
from learning.models import NN
from learning.eval import eval
from argparse import ArgumentParser
import sys
import os


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
    k = pk.kite(initial_position, initial_velocity, args.wind)
    
    if args.alg == 'sarsa':
        Q = np.ones((n_attack, n_bank, n_beta, 3, 3))*0
        Q_traj, Q, durations, rewards = sarsa(k, Q, args, initial_position, initial_velocity)
        np.save(path + "best_quality", Q)
        np.save(path + "quality_traj", Q_traj)
    else:
        net = NN()
        net, durations, rewards= dql(k, net, args, initial_position, initial_velocity)
        torch.save(net.state_dict(), path + "best_weights.h5")


    with open(path + "return.txt", "w") as file:
        for i in range(len(durations)):
            file.write(str(durations[i]) + "\t" + str(rewards[i]) + "\n")
    
    eval(args, k)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="./results/const/")
    parser.add_argument("--alg", default="sarsa")
    parser.add_argument("--wind", default="const") #const, lin or turbo
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
    args = parser.parse_args()
    main(args)
