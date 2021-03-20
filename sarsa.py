import numpy as np
import pykite as pk

n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=10
max_power=8000
eta=1
gamma=1
eps=0.3
episode_duration=60
learning_step=0.2
horizon=episode_duration/learning_step
integration_step=0.001
integration_steps_per_learning_step=learning_step/integration_step

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

def eps_greedy_policy(Q, state):
    if np.random.rand() < 1-eps:
        A_t=greedy_action(Q, S_t)
    else:
        A_t=(np.random.randint(3), np.random.randint(3))
    return A_t

def step(Q, S_t, A_t, R_t1, S_t1, A_t1):
    Q[S_t+A_t]=Q[S_t+A_t]+eta*(R_t1+gamma*Q[S_t1+A_t1]-Q[S_t+A_t])
    return Q

def terminal_step(Q, S_t, A_t, R_t1):
    Q[S_t+A_t]=Q[S_t+A_t]+eta*(R_t1-Q[S_t+A_t])
    return Q

Q=np.ones((n_attack, n_bank, n_beta, 3, 3))
Q*=(max_power*2)


initial_position=pk.vect(np.pi/3, np.pi/24, 10)
initial_velocity=pk.vect(0, 0, 0)
wind=pk.vect(10,0,0)
k=pk.kite(initial_position, initial_velocity)
initial_beta=k.beta(wind)
S_t=(10,5,initial_beta)
print("Initial state: ", S_t)

A_t=eps_greedy_policy(Q, S_t)
print("Initial action: ", A_t)

for i in range(int(horizon)):
    new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
    if not k.evolve_system(new_attack_angle, new_bank_angle, int(integration_steps_per_learning_step), integration_step, wind):
        R_t1 = -np.inf
        print(i)
        Q=terminal_step(Q, S_t, A_t, R_t1)
        break
    S_t1 = (new_attack_angle, new_bank_angle, k.beta(wind))
    print("New state: ", S_t1)
    R_t1 = k.reward(new_attack_angle, new_bank_angle, wind)
    print("Reward: ", R_t1)
    A_t1=eps_greedy_policy(Q, S_t1)
    print("New action: ", A_t1)
    Q=step(Q, S_t, A_t, R_t1, S_t1, A_t1)
    S_t=S_t1
    A_t=A_t1
