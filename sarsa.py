import numpy as np
import pykite as pk
from learning_utils import *
import matplotlib.pyplot as plt

n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=pk.n_beta
max_power=8000
eta0=0.1
gamma=1
eps0=0.01
episode_duration=180
learning_step=0.2
horizon=int(episode_duration/learning_step)
integration_step=0.001
integration_steps_per_learning_step=int(learning_step/integration_step)

Q=np.ones((n_attack, n_bank, n_beta, 3, 3))
Q*=(max_power*2)

durations=[]
rewards=[]
episodes=6000
t=0
for j in range(episodes):
    cumulative_reward=0
    initial_position=pk.vect(np.pi/6, 0, 50)
    initial_velocity=pk.vect(0, 0, 0)
    wind=pk.vect(5,0,0)
    k=pk.kite(initial_position, initial_velocity)
    initial_beta=k.beta(wind)
    S_t=(14,3,initial_beta)
    A_t=eps_greedy_policy(Q, S_t, eps0)
    for i in range(horizon):
        t+=1
        eps=scheduling(eps0, t, 100000)
        eta=scheduling(eta0, t, 500000)
        new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
        still_alive=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step, wind)
        if not still_alive:
            R_t1 = scheduling(-300000, i, horizon/4)
            cumulative_reward+=R_t1
            print(j, "Simulation failed at learning step: ", i, " reward ", cumulative_reward)
            rewards.append(cumulative_reward)
            durations.append(i)
            Q=terminal_step(Q, S_t, A_t, R_t1, eta)
            break
        S_t1 = (new_attack_angle, new_bank_angle, k.beta(wind))
        R_t1 = k.reward(new_attack_angle, new_bank_angle, wind)
        cumulative_reward+=R_t1
        A_t1=eps_greedy_policy(Q, S_t1, eps)
        if i==int(horizon)-1:
            Q=terminal_step(Q, S_t, A_t, R_t1, eta)
            print(j, "Simulation ended at learning step: ", i, " reward ", cumulative_reward)
            rewards.append(cumulative_reward)
            durations.append(i)
        else:
            Q=step(Q, S_t, A_t, R_t1, S_t1, A_t1, eta, gamma)
        S_t=S_t1
        A_t=A_t1
theta=[]
phi=[]
r=[]
initial_position=pk.vect(np.pi/3, 0, 50)
initial_velocity=pk.vect(0, 0, 0)
wind=pk.vect(5,0,0)
k=pk.kite(initial_position, initial_velocity)
initial_beta=k.beta(wind)
S_t=(14,3,initial_beta)
A_t=eps_greedy_policy(Q, S_t, 0)
for i in range(horizon):
    theta.append(k.position.theta)
    phi.append(k.position.phi)
    r.append(k.position.r)
    new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
    still_alive=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step, wind)
    if not still_alive:
        print(j, "Simulation failed at learning step: ", i)
        print(k.position.theta, k.position.phi, k.position.r)
        break
    S_t = (new_attack_angle, new_bank_angle, k.beta(wind))
    A_t=eps_greedy_policy(Q, S_t1, 0)
    if i==int(horizon)-1:
        print(j, "Simulation ended at learning step: ", i)

theta=np.array(theta)
phi=np.array(phi)
r=np.array(r)
plot_trajectory(theta, phi, r)

plt.figure()
plt.plot(durations, 'o')
plt.plot(np.convolve(durations, np.ones(300), 'valid') / 300)
plt.show()
plt.figure()
plt.plot(rewards, 'o')
plt.plot(np.convolve(rewards, np.ones(300), 'valid') / 300)
plt.show()
