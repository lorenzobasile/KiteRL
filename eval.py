import numpy as np
from learning.algorithms import *
from learning.models import NN
from argparse import ArgumentParser
import sys

def main(args):
    path=args.path
    if args.alg=='sarsa':
        Q=np.load(path + "best_quality.npy")
    else:
        torch.set_printoptions(threshold=10_000)
        net=NN()
        net.load_state_dict(torch.load(path + "best_weights.h5"))
        net.eval()
        for p in net.named_parameters():
            print(p)
    t=0
    durations=[]
    rewards=[]

    episode_duration=args.duration
    learning_step=0.2
    horizon=int(episode_duration/learning_step)
    integration_step=0.001
    integration_steps_per_learning_step=int(learning_step/integration_step)
    wind_type=args.wind
    penalty=1
    episodes=int(args.episodes)

    n_attack=pk.coefficients.shape[0]
    n_bank=pk.bank_angles.shape[0]
    n_beta=pk.n_beta

    initial_position=pk.vect(np.pi/6, 0, 50)
    initial_velocity=pk.vect(0, 0, 0)
    k=pk.kite(initial_position, initial_velocity, wind_type)

    r = []
    theta = []
    phi = []
    alpha = []
    bank = []
    beta = []

    ep=0
    for ep in range(episodes):
        cumulative_reward=0
        k.reset(initial_position, initial_velocity, wind_type)
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
            if args.alg=='dql':
                tensor_state=torch.tensor(S_t[0:2]).float()
                tensor_state[0]-=(n_attack/2)
                tensor_state[1]-=(n_bank/2)
                tensor_state[0]/=n_attack
                tensor_state[1]/=n_bank
                tensor_state=torch.nn.functional.one_hot(torch.tensor([S_t[0]*n_bank+S_t[1]]), num_classes=n_bank*n_attack).float().reshape(-1)
                q=net(tensor_state).reshape(3,3)
                A_t=greedy_action(q.detach().numpy())
                A_t=best_policy(S_t[0:2])
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
            if i==int(horizon)-1 or k.position.r>100:
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="./results/")
    parser.add_argument("--alg", default="sarsa")
    parser.add_argument("--wind", default="const") #const, lin or turbo
    parser.add_argument("--episodes", type=int, default=1e1)
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()
    main(args)
