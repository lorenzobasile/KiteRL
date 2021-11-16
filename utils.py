import pykite as pk
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torch
from mpl_toolkits.mplot3d import Axes3D

n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=pk.n_beta


def apply_action(state, action):
    if action==(1,1):
        return state[0],state[1]
    elif action==(0,1):
        return max(state[0]-1, 0), state[1]
    elif action==(1,0):
        return state[0], max(state[1]-1, 0)
    elif action==(0,0):
        return max(state[0]-1, 0), max(state[1]-1, 0)
    elif action==(2,1):
        return min(state[0]+1, n_attack-1), state[1]
    elif action==(1,2):
        return state[0], min(state[1]+1, n_bank-1)
    elif action==(2,2):
        return min(state[0]+1, n_attack-1), min(state[1]+1, n_bank-1)
    elif action==(0,2):
        return max(state[0]-1, 0), min(state[1]+1, n_bank-1)
    elif action==(2,0):
        return min(state[0]+1, n_attack-1), max(state[1]-1, 0)


def greedy_action(Q, state):
    return np.unravel_index(np.argmax(Q[state]), Q[state].shape)

def eps_greedy_policy(Q, state, eps):
    if np.random.rand() < 1-eps:
        A_t=greedy_action(Q, state)
    else:
        A_t=(np.random.randint(3), np.random.randint(3))
    return A_t

def step(Q, S_t, A_t, R_t1, S_t1, A_t1, eta, gamma):
    Q[S_t+A_t]=Q[S_t+A_t]+eta*(R_t1+gamma*Q[S_t1+A_t1]-Q[S_t+A_t])
    return Q

def scheduling(value, t, T, exp=0.6):
    if t>T:
        return value/((t-T)**exp)
    else:
        return value

def scheduling_c(value, t, T, exp=0.6, ac=1):
    if t>T:
        return value*ac/(ac-(t-T)**exp)
    else:
        return value

def terminal_step(Q, S_t, A_t, R_t1, eta):
    Q[S_t+A_t]=Q[S_t+A_t]+eta*(R_t1-Q[S_t+A_t])
    return Q

def plot_trajectory(theta, phi, r, save=None, marker='-'):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    x=np.multiply(r, np.multiply(np.sin(theta), np.cos(phi)))
    y=np.multiply(r, np.multiply(np.sin(theta), np.sin(phi)))
    z=np.multiply(r, np.cos(theta))
    line,=ax.plot(x, y, z, marker)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if save is not None:
        plt.savefig(save)
    plt.show()

def dql(k, net, params, initial_position, initial_velocity):
    durations=[]
    rewards=[]
    losses=[]
    lr=params['eta0']
    if params['optimizer']=='adam':
        optimizer=torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.000)
    else:
        optimizer=torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.000)
    if params['loss']=='huber':
        loss=torch.nn.SmoothL1Loss()
    else:
        loss=torch.nn.MSELoss()
    t=0
    wind_type=params['wind_type']
    episodes=int(params['episodes'])
    for episode in range(episodes):
        print(episode)
        k.reset(initial_position, initial_velocity, wind_type)
        duration, reward=dql_episode(k, net, optimizer, loss, params, initial_position, initial_velocity, t)
        t+=duration
        durations.append(duration)
        rewards.append(reward)
    return net, durations, rewards

def dql_episode(k, net, optimizer, loss, params, initial_position, initial_velocity, t):
    episode_duration=params['episode_duration']
    learning_step=params['learning_step']
    horizon=int(episode_duration/learning_step)
    integration_step=params['integration_step']
    integration_steps_per_learning_step=int(learning_step/integration_step)
    eta0=params['eta0']
    gamma=params['gamma']
    eps0=params['eps0']
    eps_exp=params['eps_decay_rate']
    eps_start=params['eps_decay_start']
    eta_exp=params['eta_decay_rate']
    eta_start=params['eta_decay_start']
    cumulative_reward=0
    factor=1e5
    initial_beta=k.beta()
    S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), initial_beta)
    k.C_l, k.C_d = pk.coefficients[S_t[0],0], pk.coefficients[S_t[0],1]
    k.psi = np.deg2rad(pk.bank_angles[S_t[1]])
    acc=k.accelerations()
    S_t+=acc
    tensor_state=torch.tensor(S_t).float()
    tensor_state[0]/=n_attack
    tensor_state[1]/=n_bank
    tensor_state[2]/=n_beta
    for i in range(horizon):
        t+=1
        eps=scheduling(eps0, t, eps_start, exp=eps_exp)
        if i<300:
            eps=eps*1
        else:
            eps=eps*5
        q=net(tensor_state).reshape(3,3)
        A_t=torch.randint(3,(2,)) if np.random.rand()<eps else (q==torch.max(q)).nonzero().reshape(-1)
        A_t=A_t[0],A_t[1]
        optimizer.param_groups[0]['lr']=scheduling(eta0, t, eta_start, exp=eta_exp)
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        if not sim_status==0:
            R_t1 = scheduling(-300000.0/factor, i, horizon/4)
            cumulative_reward+=R_t1
            target=torch.tensor(R_t1)
            l=loss(target, q[A_t])
            print("epsilon ", eps, " eta", optimizer.param_groups[0]['lr'], "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            break
        S_t1 = (new_attack_angle, new_bank_angle, k.beta())
        acc=k.accelerations()
        S_t1+=acc
        tensor_state=torch.tensor(S_t1).float()
        tensor_state[0]/=n_attack
        tensor_state[1]/=n_bank
        tensor_state[2]/=n_beta
        R_t1 = k.reward(learning_step)/factor
        cumulative_reward+=R_t1
        if i==int(horizon)-1:
            print("Simulation ended at learning step: ", i, " reward ", cumulative_reward)
            target=torch.tensor(R_t1)
        else:
            target=R_t1+gamma*torch.max(net(tensor_state))
        l=loss(target, q[A_t])
        S_t=S_t1
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    return cumulative_reward, i+1

def sarsa(k, Q, params, initial_position, initial_velocity):
    t=0
    durations=[]
    rewards=[]
    eta0=params['eta0']
    gamma=params['gamma']
    eps0=params['eps0']
    eps_exp=params['eps_decay_rate']
    eps_start=params['eps_decay_start']
    eta_exp=params['eta_decay_rate']
    eta_start=params['eta_decay_start']
    episode_duration=params['episode_duration']
    learning_step=params['learning_step']
    horizon=int(episode_duration/learning_step)
    integration_step=params['integration_step']
    integration_steps_per_learning_step=int(learning_step/integration_step)
    wind_type=params['wind_type']
    episodes=int(params['episodes'])
    for j in range(episodes):
        cumulative_reward=0
        k.reset(initial_position, initial_velocity, wind_type)
        initial_beta=k.beta()
        S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), initial_beta)
        k.C_l, k.C_d = pk.coefficients[S_t[0],0], pk.coefficients[S_t[0],1]
        k.psi = np.deg2rad(pk.bank_angles[S_t[1]])
        A_t=eps_greedy_policy(Q, S_t, eps0)
        for i in range(horizon):
            t+=1
            eps=scheduling(eps0, t, eps_start, exp=eps_exp)
            eta=scheduling(eta0, t, eta_start, exp=eta_exp)
            new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
            sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
            if not sim_status==0:
                R_t1 = scheduling(-3000000, i, horizon/4)
                cumulative_reward+=R_t1
                print(j, "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
                rewards.append(cumulative_reward)
                durations.append(i+1)
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
                durations.append(i+1)
            else:
                Q=step(Q, S_t, A_t, R_t1, S_t1, A_t1, eta, gamma)
            S_t=S_t1
            A_t=A_t1
    return Q, durations, rewards
