import numpy as np
from learning.algorithms import *
from learning.models import NN
import matplotlib.pyplot as plt
import learning.utils as ut

learning_step=0.2
integration_step=0.001
penalty=0.1

def eval(args, k):
    with torch.no_grad():
        path=args.path

        d_traj, r_traj = ut.read_traj(path+'/return.txt')
        '''
        lr=np.ones_like(r_traj)
        eps=np.ones_like(r_traj)
        for i in range(len(r_traj)):
            nsteps=np.sum(d_traj[:i])
            lr[i]=scheduling(args.lr, nsteps, args.lrstart, args.lrrate)
            eps[i]=scheduling(args.eps, nsteps, args.epsstart, args.epsrate)
        '''
        plt.figure(figsize=(10,6))
        plt.title("Cumulative reward")
        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('kWh', fontsize=16)
        plt.plot(r_traj, 'o')
        #plt.plot(lr, label='lr')
        #plt.plot(eps, label='eps')
        smooth = np.convolve(r_traj, np.ones(100), "valid")/100
        plt.plot(smooth, color='red', lw=1)
        #plt.legend()
        plt.savefig(path+'return.png', dpi=200)
        plt.show()

        
        plt.figure(figsize=(10,6))
        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('Durations', fontsize=16)
        plt.plot(d_traj, 'o')
        smooth = np.convolve(d_traj, np.ones(100), "valid")/100
        plt.plot(smooth, color='red', lw=1)
        plt.savefig(path+'durations.png', dpi=200)
        plt.show()

        if args.alg=='sarsa':

            Q_traj = np.load(path+"quality_traj.npy")
            Q_traj = Q_traj.reshape(Q_traj.shape[0], -1)
            stop=np.where(np.max(Q_traj[1:], axis=1)==0)[0][0]
            print(stop)
            plt.figure(figsize = (15,10))
            for i in range(Q_traj.shape[1]):
                plt.plot(Q_traj[:stop,i], '-')
            plt.ylabel("Q(s,a)")
            plt.xlabel("Learning step")
            plt.savefig(path+"quality_traj.png")
            plt.show()



        if args.alg=='sarsa':
            Q=np.load(path + "best_quality.npy")
        else:
            torch.set_printoptions(threshold=10_000)
            net=NN()
            net.load_state_dict(torch.load(path + "best_weights.h5"))
            net.eval()
        t=0
        durations=[]
        rewards=[]

        episode_duration=args.duration

        horizon=int(episode_duration/learning_step)

        integration_steps_per_learning_step=int(learning_step/integration_step)
        wind_type=args.wind
        episodes=int(args.eval_episodes)

        n_attack=pk.coefficients.shape[0]
        n_bank=pk.bank_angles.shape[0]
        n_beta=pk.n_beta

        initial_position=pk.vect(np.pi/6, 0, 20)
        initial_velocity=pk.vect(0, 0, 0)
        #k=pk.kite(initial_position, initial_velocity, wind_type)

        r = []
        theta = []
        phi = []
        alpha = []
        bank = []
        beta = []

        ep=0
        for ep in range(episodes):
            cumulative_reward=0
            #initial_position=pk.vect(np.pi/6+np.random.normal(0,0.04), 0, 20+np.random.normal(0,1))
            k.reset(initial_position, initial_velocity, wind_type)
            initial_beta=k.beta(continuous=(args.alg=='dql'))
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
                    tensor_state=torch.tensor(S_t)
                    tensor_state[0]-=(n_attack/2)
                    tensor_state[1]-=(n_bank/2)
                    tensor_state[0]/=n_attack
                    tensor_state[1]/=n_bank
                    tensor_state[2]/=(np.pi/2)
                    q=net(tensor_state).reshape(3,3)
                    A_t=greedy_action(q.detach().numpy())
                else:
                    A_t=greedy_action(Q[S_t])
                t+=1
                new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
                sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
                r.append(k.position.r)
                theta.append(k.position.theta)
                phi.append(k.position.phi)
                S_t1 = (new_attack_angle, new_bank_angle, k.beta(continuous=(args.alg=='dql')))
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
                if i==int(horizon)-1 or k.fullyunrolled():
                    print(ep, "Simulation ended at learning step: ", i, " reward ", cumulative_reward)
                    rewards.append(cumulative_reward)
                    durations.append(i+1)
                    break
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
        durations=np.array(durations)
        rewards=np.array(rewards)
        coordinates = np.load(path+"eval_traj.npy")
        controls = np.load(path+"contr_traj.npy")
        print(coordinates[0])

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3.5))

        lim = [0,1000]

        ax1.set_xlim(lim)
        ax1.set_xlabel('Time, seconds', fontsize=14)
        ax1.set_ylabel('Kite x, meters', fontsize=14)
        ax1.plot(coordinates[:,0])

        ax2.set_xlim(lim)
        ax2.set_xlabel('Time, seconds', fontsize=14)
        ax2.set_ylabel('Kite y, meters', fontsize=14)
        ax2.plot(coordinates[:,1])

        ax3.set_xlim(lim)
        ax3.set_xlabel('Time, seconds', fontsize=14)
        ax3.set_ylabel('Kite z, meters', fontsize=14)
        ax3.plot(coordinates[:,2])

        plt.tight_layout()
        plt.savefig(path+"eval_traj.png", dpi=200)
        plt.show()
