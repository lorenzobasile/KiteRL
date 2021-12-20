import learning.pykite as pk
import numpy as np
import torch
from copy import deepcopy
from learning.experiencereplay import ExperienceBuffer
from time import time

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


def greedy_action(Q_values):
    return np.unravel_index(np.argmax(Q_values), Q_values.shape)

def eps_greedy_policy(Q_values, eps):
    if np.random.rand() < 1-eps:
        A_t=greedy_action(Q_values)
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
        return value*ac/(ac+(t-T)**exp)
    else:
        return value

def terminal_step(Q, S_t, A_t, R_t1, eta):
    Q[S_t+A_t]=Q[S_t+A_t]+eta*(R_t1-Q[S_t+A_t])
    return Q

def dql(k, net, params, initial_position, initial_velocity):
    durations=[]
    rewards=[]
    losses=[]
    lr=params['eta0']
    buffer_size=params['buffer_size']
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
    if 'episodes' in params:
        episodes=int(params['episodes'])
        max_steps=int(params['episodes']*params['episode_duration']/params['learning_step'])
    else:
        max_steps=int(params['max_steps'])
    Q_traj = np.zeros(((max_steps//1000+1,)+(n_attack, n_bank, 3, 3)))
    l_traj = np.zeros((episodes,))
    w = 0
    experience_buffer=ExperienceBuffer(buffer_size, 6, 2)
    visits=np.zeros((n_attack, n_bank, n_beta, 3, 3), dtype='int')
    for episode in range(episodes):
        print(episode)
        k.reset(initial_position, initial_velocity, wind_type, params)
        duration, reward, experience_buffer, Q_traj, l, w, visits = dql_episode(k, net, optimizer, experience_buffer, loss, params, initial_position, initial_velocity, t, Q_traj, w, visits)
        l_traj[episode] = l
        t+=duration
        durations.append(duration)
        rewards.append(reward)
    return net, durations, rewards, Q_traj, l_traj

def dql_episode(k, net, optimizer, experience_buffer, loss, params, initial_position, initial_velocity, t, Q_traj, w, visits):
    target_net=deepcopy(net)
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
    eps_c=params['eps_c']
    eta_exp=params['eta_decay_rate']
    eta_start=params['eta_decay_start']
    eta_c=params['eta_c']
    penalty=params['penalty']
    batch_size=params['batch_size']
    cumulative_reward=0
    S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), k.beta())
    k.update_coefficients(S_t[0], S_t[1])
    acc=k.accelerations()
    S_t+=acc
    tensor_state=torch.tensor(S_t).float()
    for i in range(horizon):
        t0=time()
        t+=1
        eps=scheduling_c(eps0, t, eps_start, exp=eps_exp, ac=eps_c)
        q=net(tensor_state).reshape(3,3)
        A_t=eps_greedy_policy(q.detach().numpy(), eps)
        optimizer.param_groups[0]['lr']=scheduling_c(eta0, t, eta_start, exp=eta_exp, ac=eta_c)
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        t1=time()
        sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        t2=time()
        if not sim_status==0:
            R_t1 = scheduling(-penalty, i, horizon)
            experience_buffer.insert(torch.cat([tensor_state, torch.tensor(A_t), torch.tensor(R_t1).reshape(1), tensor_state, torch.tensor(0).reshape(1)]))
            cumulative_reward+=R_t1
            target=torch.tensor(R_t1)
            l=compute_loss(loss, experience_buffer.get_batch(batch_size), net)
            print("epsilon ", eps, " eta", optimizer.param_groups[0]['lr'], "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
            optimizer.zero_grad()
            l.backward()
            if experience_buffer.current_size>batch_size:
                optimizer.step()
            break
        S_t1 = (new_attack_angle, new_bank_angle, k.beta())
        acc=k.accelerations()
        S_t1+=acc
        new_tensor_state=torch.tensor(S_t1).float()
        R_t1 = k.reward(learning_step)
        cumulative_reward+=R_t1
        if i==int(horizon)-1:
            experience_buffer.insert(torch.cat([tensor_state, torch.tensor(A_t), torch.tensor(R_t1).reshape(1), new_tensor_state, torch.tensor(0).reshape(1)]))
            print("Simulation ended at learning step: ", i, " reward ", cumulative_reward)
        else:
            experience_buffer.insert(torch.cat([tensor_state, torch.tensor(A_t), torch.tensor(R_t1).reshape(1), new_tensor_state, torch.tensor(1).reshape(1)]))
        l=compute_loss(loss, experience_buffer.get_batch(batch_size), net)
        S_t=S_t1
        optimizer.zero_grad()
        l.backward()
        if experience_buffer.current_size>batch_size:
            optimizer.step()
        tensor_state=new_tensor_state
        '''
        if (t-1)%1000 == 0:
            Q = np.zeros((n_attack, n_bank, 3, 3))
            for attack in range(n_attack):
                for bank in range(n_bank):
                    attack_f=attack-(n_attack/2)
                    bank_f=bank-(n_bank/2)
                    attack_f /= n_attack
                    bank_f /= n_bank
                    Q[attack][bank] = np.array(net(torch.tensor([attack_f, bank_f])).reshape(3,3).detach().numpy())
            Q_traj[w] = Q
            w+=1
        '''
    return i+1, cumulative_reward, experience_buffer, Q_traj, l, w, visits

def reshape_actions(actions):
    oned_actions=torch.zeros(actions.shape[0])
    oned_actions=actions[:,0]*3+actions[:,1]
    return oned_actions

def compute_loss(loss, batch, net):
    s=batch[:,:6]
    a=batch[:,6:8]
    r=batch[:,8]
    a=reshape_actions(a)
    sprime=batch[:,9:15]
    notfinished=batch[:,15]
    with torch.no_grad():
        target=r+1*torch.max(net(sprime)).detach()*notfinished
    return loss(target, torch.gather(net(s), 1, a.long().reshape(-1,1)).reshape(-1))


def sarsa(k, Q, params, initial_position, initial_velocity):
    t=0
    w=0
    visits=np.zeros_like(Q, dtype='int')
    durations=[]
    rewards=[]
    eta0=params['eta0']
    gamma=params['gamma']
    eps0=params['eps0']
    eps_exp=params['eps_decay_rate']
    eps_start=params['eps_decay_start']
    eps_c=params['eps_c']
    eta_exp=params['eta_decay_rate']
    eta_start=params['eta_decay_start']
    eta_c=params['eta_c']
    episode_duration=params['episode_duration']
    learning_step=params['learning_step']
    horizon=int(episode_duration/learning_step)
    integration_step=params['integration_step']
    integration_steps_per_learning_step=int(learning_step/integration_step)
    wind_type=params['wind_type']
    penalty=params['penalty']
    if 'episodes' in params:
        episodes=int(params['episodes'])
        max_steps=int(params['episodes']*params['episode_duration']/params['learning_step'])
    else:
        max_steps=int(params['max_steps'])
    Q_traj = np.zeros(((max_steps//1000+1,)+Q.shape))
    ep=0
    while ep<=episodes if 'episodes' in params else t<max_steps:
    #for j in range(episodes):
        cumulative_reward=0
        ep+=1
        k.reset(initial_position, initial_velocity, wind_type, params)
        initial_beta=k.beta()
        S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), initial_beta)
        k.update_coefficients(S_t[0], S_t[1])
        A_t=eps_greedy_policy(Q[S_t], eps0)
        for i in range(horizon):
            visits[S_t+A_t]+=1
            t+=1
            eps=scheduling_c(eps0, t, eps_start, exp=eps_exp, ac=eps_c)
            eta=scheduling_c(eta0, visits[S_t+A_t], eta_start, exp=eta_exp, ac=eta_c)
            new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
            sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
            if not sim_status==0:
                R_t1 = scheduling(-penalty, i, horizon)
                cumulative_reward+=R_t1
                print(ep, "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
                rewards.append(cumulative_reward)
                durations.append(i+1)
                Q=terminal_step(Q, S_t, A_t, R_t1, eta)
                break
            S_t1 = (new_attack_angle, new_bank_angle, k.beta())
            R_t1 = k.reward(learning_step)
            cumulative_reward+=R_t1
            A_t1=eps_greedy_policy(Q[S_t1], eps)
            if k.position.r>950 or i==int(horizon)-1:
                Q=terminal_step(Q, S_t, A_t, R_t1, eta)
                print(ep, "Simulation ended at learning step: ", i, " reward ", cumulative_reward)
                rewards.append(cumulative_reward)
                durations.append(i+1)
                break
            else:
                Q=step(Q, S_t, A_t, R_t1, S_t1, A_t1, eta, gamma)
            S_t=S_t1
            A_t=A_t1
            if (t-1)%1000 == 0:
                Q_traj[w] = Q
                w+=1

    return Q_traj, Q, durations, rewards
