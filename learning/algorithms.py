import learning.pykite as pk
import numpy as np
import torch
from copy import deepcopy
from learning.experiencereplay import ExperienceBuffer

n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=pk.n_beta
n_states=3
n_actions=2
penalty=0.1
integration_step=0.001
learning_step=0.2


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

def step(Q, S_t, A_t, R_t1, S_t1, A_t1, eta):
    Q[S_t+A_t]=Q[S_t+A_t]+eta*(R_t1+Q[S_t1+A_t1]-Q[S_t+A_t])
    return Q

def scheduling(value, t, T, exp=0.6):
    if t>T:
        return value/((t-T)**exp)
    else:
        return value

def terminal_step(Q, S_t, A_t, R_t1, eta):
    Q[S_t+A_t]=Q[S_t+A_t]+eta*(R_t1-Q[S_t+A_t])
    return Q

def dql(k, net, args, initial_position, initial_velocity):
    durations=[]
    rewards=[]
    losses=[]
    lr=args.lr
    buffer_size=1
    optimizer=torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.00)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 3000)
    loss=torch.nn.SmoothL1Loss()
    t=0
    wind_type=args.wind
    episodes=int(args.episodes)
    experience_buffer=ExperienceBuffer(buffer_size, n_states, n_actions)
    for episode in range(episodes):
        print(episode)
        k.reset(initial_position, initial_velocity, wind_type)
        duration, reward, experience_buffer, net = dql_episode(k, net, optimizer, experience_buffer, loss, args, initial_position, initial_velocity, t)
        scheduler.step()
        t+=duration
        durations.append(duration)
        rewards.append(reward)
    return net, durations, rewards

def dql_episode(k, net, optimizer, experience_buffer, loss, args, initial_position, initial_velocity, t):
    target_net=deepcopy(net)
    episode_duration=args.duration
    horizon=int(episode_duration/learning_step)
    integration_steps_per_learning_step=int(learning_step/integration_step)
    eta0=args.lr
    eps0=args.eps
    eps_exp=args.epsrate
    eps_start=args.epsstart
    eta_exp=args.lrrate
    eta_start=args.lrstart
    batch_size=1
    cumulative_reward=0
    S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), k.beta(continuous=True))
    k.update_coefficients(S_t[0], S_t[1])
    tensor_state=torch.tensor(S_t).float()
    tensor_state[0]-=(n_attack/2)
    tensor_state[1]-=(n_bank/2)
    tensor_state[0]/=n_attack
    tensor_state[1]/=n_bank
    tensor_state[2]/=(np.pi/2)
    for i in range(horizon):
        target_net=deepcopy(net) #to DISABLE double ql
        t+=1
        eps=scheduling(eps0, t, eps_start, exp=eps_exp)
        q=net(tensor_state).reshape(3,3)
        A_t=eps_greedy_policy(q.detach().numpy(), eps)
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        if not sim_status==0:
            R_t1 = scheduling(-penalty, i, horizon)
            experience_buffer.insert(torch.cat([tensor_state, torch.tensor(A_t), torch.tensor(R_t1).reshape(1), tensor_state, torch.tensor(0).reshape(1)]))
            cumulative_reward+=R_t1
            target=torch.tensor(R_t1)
            l=compute_loss(loss, experience_buffer.get_batch(batch_size), net, target_net)
            print("epsilon ", eps, " eta", optimizer.param_groups[0]['lr'], "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
            optimizer.zero_grad()
            l.backward()
            if experience_buffer.current_size>=batch_size:
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
            break
        S_t1 = (new_attack_angle, new_bank_angle, k.beta(continuous=True))
        new_tensor_state=torch.tensor(S_t1).float()
        new_tensor_state[0]-=(n_attack/2)
        new_tensor_state[1]-=(n_bank/2)
        new_tensor_state[0]/=n_attack
        new_tensor_state[1]/=n_bank
        new_tensor_state[2]/=(np.pi/2)
        R_t1 = k.reward(learning_step)
        cumulative_reward+=R_t1
        A_t1=eps_greedy_policy(target_net(new_tensor_state).detach().numpy(), eps)
        if i==int(horizon)-1 or k.fullyunrolled():
            target=torch.tensor(R_t1)
            experience_buffer.insert(torch.cat([tensor_state, torch.tensor(A_t), torch.tensor(R_t1).reshape(1), new_tensor_state, torch.tensor(0).reshape(1)]))
            print("Simulation ended at learning step: ", i, " reward ", cumulative_reward)
            break
        else:
            with torch.no_grad():
                target=R_t1+target_net(new_tensor_state).detach().reshape(3,3)[best_policy(S_t1)]
            experience_buffer.insert(torch.cat([tensor_state, torch.tensor(A_t), torch.tensor(R_t1).reshape(1), new_tensor_state, torch.tensor(1).reshape(1)]))
        l=compute_loss(loss, experience_buffer.get_batch(batch_size), net, target_net)
        S_t=S_t1
        optimizer.zero_grad()
        l.backward()
        if experience_buffer.current_size>=batch_size:
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        tensor_state=new_tensor_state
    return i+1, cumulative_reward, experience_buffer, net

def reshape_actions(actions):
    oned_actions=torch.zeros(actions.shape[0])
    oned_actions=actions[:,0]*3+actions[:,1]
    return oned_actions

def compute_loss(loss, batch, net, target_net):
    s=batch[:,:n_states]
    a=batch[:,n_states:n_states+n_actions]
    r=batch[:,n_states+n_actions]
    a=reshape_actions(a)
    sprime=batch[:,n_states+n_actions+1:-1]
    notfinished=batch[:,-1]
    with torch.no_grad():
        target=r+torch.max(target_net(sprime)).detach()*notfinished
    return loss(target, torch.gather(net(s), 1, a.long().reshape(-1,1)).reshape(-1))

def sarsa(k, Q, args, initial_position, initial_velocity):
    t=0
    w=0
    visits=np.zeros_like(Q, dtype='int')
    durations=[]
    rewards=[]
    eta0=args.lr
    eps0=args.eps
    eps_exp=args.epsrate
    eps_start=args.epsstart
    eta_exp=args.lrrate
    eta_start=args.lrstart
    episode_duration=int(args.duration)
    horizon=int(episode_duration/learning_step)
    integration_steps_per_learning_step=int(learning_step/integration_step)
    wind_type=args.wind
    episodes=int(args.episodes)
    Q_traj = np.zeros(((int(episodes*episode_duration/learning_step)//10000+1,)+Q.shape))
    for ep in range(episodes):
        cumulative_reward=0
        k.reset(initial_position, initial_velocity, wind_type)
        initial_beta=k.beta()
        S_t=(np.random.randint(3,n_attack), np.random.randint(0,n_bank), initial_beta)
        initial_state=S_t
        k.update_coefficients(S_t[0], S_t[1])
        A_t=eps_greedy_policy(Q[S_t], eps0)
        for i in range(horizon):
            visits[S_t+A_t]+=1
            t+=1
            eps=scheduling(eps0, t, eps_start, exp=eps_exp)
            if args.personalizedlr:
                eta=scheduling(eta0, visits[S_t+A_t], eta_start, exp=eta_exp)
            else:
                eta=scheduling(eta0, t, eta_start, exp=eta_exp)
            new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
            sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
            if not sim_status==0:
                R_t1 = scheduling(-penalty, i, horizon)
                cumulative_reward+=R_t1
                print(ep, "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
                print("Initial state: ", initial_state)
                rewards.append(cumulative_reward)
                durations.append(i+1)
                Q=terminal_step(Q, S_t, A_t, R_t1, eta)
                break
            S_t1 = (new_attack_angle, new_bank_angle, k.beta())
            R_t1 = k.reward(learning_step)
            cumulative_reward+=R_t1
            A_t1=eps_greedy_policy(Q[S_t1], eps)
            if i==int(horizon)-1 or k.fullyunrolled():
                Q=terminal_step(Q, S_t, A_t, R_t1, eta)
                print(ep, "Simulation ended at learning step: ", i, " reward ", cumulative_reward)
                rewards.append(cumulative_reward)
                durations.append(i+1)
                break
            else:
                Q=step(Q, S_t, A_t, R_t1, S_t1, A_t1, eta)
            S_t=S_t1
            A_t=A_t1
            if (t-1)%10000 == 0:
                Q_traj[w] = Q
                w+=1
    return Q_traj, Q, durations, rewards
