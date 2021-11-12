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

def dql(net, opt, episodes, horizon, learning_step, integration_step, integration_steps_per_learning_step, initial_pos, initial_vel, wind, lr=0.1, lr_decay_exp=0.6, lr_decay_start=200000, eps=0.01, eps_decay_exp=0.6, eps_decay_start=400000, gamma=1, plot=False):
    durations=[]
    rewards=[]
    losses=[]
    if opt=='adam':
        optimizer=torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.000)
    else:
        optimizer=torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.000)
    loss=torch.nn.SmoothL1Loss()
    t=0
    k=pk.kite(initial_pos, initial_vel)
    for episode in range(episodes):
        print(episode)
        k.reset(initial_pos, initial_vel)
        t,theta,phi,r=dql_episode(k, net, optimizer, loss, initial_pos, initial_vel, wind, horizon, learning_step, integration_step, integration_steps_per_learning_step, lr, lr_decay_exp, lr_decay_start, eps, eps_decay_exp, eps_decay_start, durations, rewards, gamma, t, plot)
    return durations, rewards,theta,phi,r

def dql_episode(k, net, optimizer, loss, initial_position, initial_velocity, wind, horizon, learning_step, integration_step, integration_steps_per_learning_step, eta0, eta_decay, eta_decay_start, eps0, eps_decay, eps_decay_start, durations, rewards, gamma, t, plot):
    if plot:
        theta=[]
        phi=[]
        r=[]
    else:
        theta=None
        phi=None
        r=None
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
        '''
        if k.position.r*np.cos(k.position.theta)>100:
            print(k.position.r*np.cos(k.position.theta))
            print(k.position.r)
            break
        '''
        if plot:
            theta.append(k.position.theta)
            phi.append(k.position.phi)
            r.append(k.position.r)
        t+=1
        eps=scheduling(eps0, t, eps_decay_start, exp=eps_decay)
        if i<300:
            eps=eps*1
        else:
            eps=eps*5
        q=net(tensor_state).reshape(3,3)
        A_t=torch.randint(3,(2,)) if np.random.rand()<eps else (q==torch.max(q)).nonzero().reshape(-1)
        A_t=A_t[0],A_t[1]
        optimizer.param_groups[0]['lr']=scheduling(eta0, t, eta_decay_start, exp=eta_decay)
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        if not sim_status==0:
            R_t1 = scheduling(-300000.0/factor, i, horizon/4)
            cumulative_reward+=R_t1
            rewards.append(cumulative_reward)
            durations.append(i)
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
            rewards.append(cumulative_reward)
            durations.append(i)
        else:
            target=R_t1+gamma*torch.max(net(tensor_state))
        l=loss(target, q[A_t])
        S_t=S_t1
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    return t,theta,phi,r

def dql_eval(net, horizon, integration_step, integration_steps_per_learning_step, initial_position, initial_velocity, wind):
    theta=[]
    phi=[]
    r=[]
    net.eval()
    k=pk.kite(initial_position, initial_velocity, wind)
    initial_beta=k.beta()
    S_t=(14,3,initial_beta)
    acc=k.accelerations()
    S_t+=acc
    for i in range(horizon):
        tensor_state=torch.tensor(S_t).float()
        tensor_state[0]/=n_attack
        tensor_state[1]/=n_bank
        tensor_state[2]/=n_beta
        q=net(tensor_state).reshape(3,3)
        print(torch.max(q))
        #print(S_t[:2])
        theta.append(k.position.theta)
        phi.append(k.position.phi)
        r.append(k.position.r)
        A_t=(q==torch.max(q)).nonzero().reshape(-1)
        A_t=A_t[0],A_t[1]
        #print(A_t)
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        if not status==0:
            print("Simulation failed at learning step: ", i)
            break
        S_t = (new_attack_angle, new_bank_angle, k.beta())
        acc=k.accelerations()
        S_t+=acc
        #visits[S_t]+=1
        if i==int(horizon)-1:
            print( "Simulation ended at learning step: ", i)
    return theta, phi, r
'''
def update_weights(target_net, net):
    for pt in target_net.named_parameters():
        for pn in net.named_parameters():
            if pt[0]==pn[0]:
                pt[1].data=pn[1].data

                
                
                
def dql_post(fixed_net, switch, net, opt, episodes, horizon, learning_step, integration_step, integration_steps_per_learning_step, initial_pos, initial_vel, wind, lr=0.1, lr_decay_exp=0.6, lr_decay_start=200000, eps=0.01, eps_decay_exp=1.2, eps_decay_start=400000, gamma=1, plot=False):
    durations=[]
    rewards=[]
    losses=[]
    if opt=='adam':
        optimizer=torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.000)
    else:
        optimizer=torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.000)
    loss=torch.nn.SmoothL1Loss()
    t=0
    k=pk.kite(initial_pos, initial_vel)
    for episode in range(episodes):
        print(episode)
        k.reset(initial_pos, initial_vel)
        t,theta,phi,r=dql_post_episode(k, fixed_net, switch, net, optimizer, loss, initial_pos, initial_vel, wind, horizon, learning_step, integration_step, integration_steps_per_learning_step, lr, lr_decay_exp, lr_decay_start, eps, eps_decay_exp, eps_decay_start, durations, rewards, gamma, t, plot)
    return durations, rewards,theta,phi,r

def dql_post_episode(k, fixed_net, switch, net, optimizer, loss, initial_position, initial_velocity, wind, horizon, learning_step, integration_step, integration_steps_per_learning_step, eta0, eta_decay, eta_decay_start, eps0, eps_decay, eps_decay_start, durations, rewards, gamma, t, plot):
    if plot:
        theta=[]
        phi=[]
        r=[]
    else:
        theta=None
        phi=None
        r=None
    factor=1e5
    cumulative_reward=0
    initial_beta=k.beta()
    S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank))
    acc=k.accelerations()
    S_t+=acc
    k.C_l, k.C_d = pk.coefficients[S_t[0],0], pk.coefficients[S_t[0],1]
    k.psi = np.deg2rad(pk.bank_angles[S_t[1]])
    tensor_state=torch.tensor(S_t).float()
    tensor_state[0]/=n_attack
    tensor_state[1]/=n_bank
    #tensor_state[2]/=n_beta
    #print(net(tensor_state).reshape(3,3))
    for i in range(horizon):
        if plot:
            theta.append(k.position.theta)
            phi.append(k.position.phi)
            r.append(k.position.r)
        t+=1
        
        eps=scheduling(eps0, t, eps_decay_start, exp=eps_decay)
        if i<switch:
            q=fixed_net(tensor_state).reshape(3,3)
            A_t=(q==torch.max(q)).nonzero().reshape(-1)
            A_t=A_t[0],A_t[1]
            new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
            sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
            if not sim_status==0:
                R_t1 = scheduling(-300000.0/factor, i, horizon/4)
                cumulative_reward+=R_t1
                rewards.append(cumulative_reward)
                durations.append(i)
                print("epsilon ", eps, " eta", optimizer.param_groups[0]['lr'], "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
                break
            S_t1 = (new_attack_angle, new_bank_angle)
            acc=k.accelerations()
            S_t1+=acc
            tensor_state=torch.tensor(S_t1).float()
            tensor_state[0]/=n_attack
            tensor_state[1]/=n_bank
            #tensor_state[2]/=n_beta
            R_t1 = k.reward(learning_step)/factor
            cumulative_reward+=R_t1
            S_t=S_t1
        else:
            
            if i<300:
                eps=eps
            else:
                eps=eps*5
            
            optimizer.param_groups[0]['lr']=scheduling(eta0, t, eta_decay_start, exp=eta_decay)
            q=net(tensor_state).reshape(3,3)
            A_t=torch.randint(3,(2,)) if np.random.rand()<eps else (q==torch.max(q)).nonzero().reshape(-1)
            A_t=A_t[0],A_t[1]
            #print(q)
            #print(A_t)
            new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
            sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
            if not sim_status==0:
                R_t1 = scheduling(-300000.0/factor, i, horizon/4)
                cumulative_reward+=R_t1
                rewards.append(cumulative_reward)
                durations.append(i)
                target=torch.tensor(R_t1)
                l=loss(target, q[A_t])
                print("epsilon ", eps, " eta", optimizer.param_groups[0]['lr'], "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                #print(net(tensor_state).reshape(3,3))
                break
            S_t1 = (new_attack_angle, new_bank_angle)
            #print(S_t1)
            acc=k.accelerations()
            S_t1+=acc
            tensor_state=torch.tensor(S_t1).float()
            tensor_state[0]/=n_attack
            tensor_state[1]/=n_bank
            #tensor_state[2]/=n_beta
            #print("new state ", tensor_state)
            R_t1 = k.reward(learning_step)/factor
            cumulative_reward+=R_t1
            if i==int(horizon)-1:
                print("Simulation ended at learning step: ", i, " reward ", cumulative_reward)
                target=torch.tensor(R_t1)
                rewards.append(cumulative_reward)
                durations.append(i)
            else:
                target=R_t1+gamma*torch.max(net(tensor_state))
            l=loss(target, q[A_t])
            S_t=S_t1
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    return t,theta,phi,r

    
    
def dql_mem(net, opt, episodes, horizon, learning_step, integration_step, integration_steps_per_learning_step, initial_pos, initial_vel, wind, lr=0.1, lr_decay_exp=0.6, lr_decay_start=200000, eps=0.01, eps_decay_exp=0.6, eps_decay_start=400000, gamma=1, plot=False):
    durations=[]
    rewards=[]
    losses=[]
    if opt=='adam':
        optimizer=torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.000)
    else:
        optimizer=torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.000)
    loss=torch.nn.SmoothL1Loss()
    t=0
    memory=torch.empty((1,15))
    k=pk.kite(initial_pos, initial_vel)
    for episode in range(episodes):
        print(episode)
        k.reset(initial_pos, initial_vel)
        t,memory,theta,phi,r=dql_episode(k, memory, net, optimizer, loss, initial_pos, initial_vel, wind, horizon, learning_step, integration_step, integration_steps_per_learning_step, lr, lr_decay_exp, lr_decay_start, eps, eps_decay_exp, eps_decay_start, durations, rewards, gamma, t, plot)
    print(memory.size())
    return durations, rewards,theta,phi,r


def dql_mem_episode(k, memory, net, optimizer, loss, initial_position, initial_velocity, wind, horizon, learning_step, integration_step, integration_steps_per_learning_step, eta0, eta_decay, eta_decay_start, eps0, eps_decay, eps_decay_start, durations, rewards, gamma, t, plot):
    if plot:
        theta=[]
        phi=[]
        r=[]
    else:
        theta=None
        phi=None
        r=None
    cumulative_reward=0
    initial_beta=k.beta()
    S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), initial_beta)
    k.C_l, k.C_d = pk.coefficients[S_t[0],0], pk.coefficients[S_t[0],1]
    k.psi = np.deg2rad(pk.bank_angles[S_t[1]])
    acc=k.accelerations()
    S_t+=acc
    tensor_state=torch.tensor(S_t).float()
    for i in range(horizon):
        if plot:
            theta.append(k.position.theta)
            phi.append(k.position.phi)
            r.append(k.position.r)
        t+=1
        eps=scheduling(eps0, t, eps_decay_start, exp=eps_decay)
        optimizer.param_groups[0]['lr']=scheduling(eta0, t, eta_decay_start, exp=eta_decay)
        q=net(tensor_state).reshape(3,3)
        A_t=torch.randint(3,(2,)) if np.random.rand()<eps else (q==torch.max(q)).nonzero().reshape(-1)
        A_tensor=A_t
        A_t=A_t[0],A_t[1]
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
        if not sim_status==0:
            R_t1 = torch.tensor(scheduling(-300000.0, i, horizon/4)).reshape(1)
            cumulative_reward+=R_t1
            rewards.append(cumulative_reward)
            durations.append(i)
            print("epsilon ", eps, " eta", optimizer.param_groups[0]['lr'], "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
            fail=True
        else:
            fail=False
            S_t1 = (new_attack_angle, new_bank_angle, k.beta())
            S_t1+=k.accelerations()
            tensor_state1=torch.tensor(S_t1).float()
            R_t1 = torch.tensor(k.reward(learning_step)).reshape(1)
            cumulative_reward+=R_t1 
        element=torch.cat((tensor_state, R_t1, A_tensor, tensor_state1)).reshape(1,15)
        if memory.size()[0]<10000:
            memory=torch.cat((memory,element))
        else:
            memory[torch.randint(10000, (1,))]=element
        if i==int(horizon)-1:
            rewards.append(cumulative_reward)
            print(cumulative_reward, i)
            durations.append(i)
        if memory.size()[0]>100:
            S,R,A,Sp=sample(memory)
            target=torch.max(R+gamma*torch.max(net(Sp)).detach(), R)
            #print(target, net(S))
            #print(net(S).gather(1, A.view(-1,1)))
            l=loss(target.reshape(32,1), net(S).gather(1, A.view(-1,1)))
            S_t=S_t1
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if fail:
            break
    return t,memory,theta,phi,r


def sample(memory):
    mem=memory[1:]
    indices=torch.randint(mem.size()[0], (32,))
    S=mem[indices,:6]
    Sp=mem[indices,-6:]
    R=mem[indices,6]
    A=(torch.mul(mem[indices,7],3)+mem[indices,8]).long()
    return S,R,A,Sp
    

'''