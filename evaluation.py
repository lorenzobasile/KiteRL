from learning.algorithms import *
from learning.models import NN
from learning.eval import eval
from argparse import ArgumentParser
import os


def main(args):
    path=args.path
    if not os.path.exists(path):
        exit()
    n_attack = pk.coefficients.shape[0]
    n_bank = pk.bank_angles.shape[0]
    n_beta = pk.n_beta
    initial_position = pk.vect(np.pi/6, 0, 20)
    initial_velocity = pk.vect(0, 0, 0)
    k = pk.kite(initial_position, initial_velocity, args.wind)
    eval(args, k)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="./results/const/")
    parser.add_argument("--alg", default="sarsa")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--eval_episodes", type=int, default=1e1)
    parser.add_argument("--wind", default="const")
    args = parser.parse_args()
    main(args)