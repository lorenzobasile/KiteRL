import os 
import argparse
import numpy as np
import learning.pykite as pk 
from learning.utils import *
import pandas as pd 
import scipy.interpolate as interpolate
from learning.models import *
from learning.algorithms import *

ACTION_NOISE = 0.2
ATTACK_INF_LIM = -5 
ATTACK_SUP_LIM = 18 
BANK_INF_LIM = -3 
BANK_SUP_LIM=3

n_attack=pk.coefficients.shape[0]
n_bank=pk.bank_angles.shape[0]
n_beta=pk.n_beta


def test(k, args): 

    if(args.range_actions) is not None: 
    
        if(len(args.range_actions[0])>1): 
        
            range_actions = np.array([args.range_actions[0][0],args.range_actions[0][2]])
            range_actions = range_actions.astype(np.float64)
            
        else: 
        
            range_actions=np.array([args.range_actions[0][0],args.range_actions[0][0]])
            range_actions = range_actions.astype(np.float64)
            
    else: 
    
        range_actions = np.array([1,1]) 
        
        
    Cl_angles = read_file('env/coefficients/CL_angle.txt') 
    
    Cl_values = read_file('env/coefficients/CL_value.txt') 
    
    Cd_angles = read_file('env/coefficients/CD_angle.txt') 
    
    Cd_values = read_file('env/coefficients/CD_value.txt') 
    
    Cl_data = pd.DataFrame({'cl_angles':Cl_angles,'cl_values':Cl_values}) 
    
    Cd_data = pd.DataFrame({'cd_angles':Cd_angles,'cd_values':Cd_values}) 
    
    f_cl  = interpolate.interp1d(Cl_data.cl_angles, Cl_data.cl_values, kind='linear')
    
    f_cd  = interpolate.interp1d(Cd_data.cd_angles, Cd_data.cd_values, kind='linear')
    
    initial_position = pk.vect(np.pi/6, 0, 20)
    
    initial_velocity = pk.vect(0, 0, 0)
    
    
    if(args.eval_episodes<25): 
    
        EPISODES=25
        
    else : 
    
        EPISODES = args.eval_episodes
        
    learning_step= args.step 
    
    integration_step = 0.001
    
    duration = 300 
    
    episode_duration= int(duration) 
    
    horizon = int(episode_duration/learning_step) 
    
    integration_steps_per_learning_step=int(learning_step/integration_step)
    
    
    dir_nets = os.path.join(args.path,"nets")
        
    
    rewards = [] 
                
    time_history = [] 
                
    KWh_per_second = []
    
    r = []
    
    theta = [] 
    
    phi = [] 

    alpha = [] 
    
    bank = [] 

    beta = []
    
    power = []
    
    cumulative_time = []
    
    int_time = 0
    if args.alg=='td3':
        agent =Agent(3,2,chkpt_dir=dir_nets)#, max_action =range_actions)
            
        agent.load_actor() 
    elif args.alg=='sarsa':
        Q=np.load(args.path + "best_quality.npy")
    
    eff_wind_speed = []
    
    for i in range(EPISODES): 
    
        done = False 
        
        time = 0 
        
        score = 0 
        
        k.reset(initial_position, initial_velocity, args.wind)
        
        initial_beta = k.beta()
                
        if k.continuous:
            S_t=(np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta)
            c_l = f_cl(S_t[0])
        
            c_d = f_cd(S_t[0])
            
            k.update_coefficients_cont(c_l,c_d,S_t[1])
        else:
            S_t=(np.random.randint(0,n_attack), np.random.randint(0,n_bank), initial_beta)
            k.update_coefficients(S_t[0], S_t[1])
        
        
        state = np.asarray(S_t) 
            
        count = 0 
                
        while(not done): 
        
            time+=learning_step
                    
            int_time += 1 
            
            if args.alg=='td3':
                action = agent.choose_action(state,0,test=True)
                new_attack, new_bank = state[0]+action[0]*range_actions[0],state[1]+action[1]*range_actions[1]
            
                new_attack = np.clip(new_attack,ATTACK_INF_LIM,ATTACK_SUP_LIM) 
                
                new_bank = np.clip(new_bank,BANK_INF_LIM,BANK_SUP_LIM)
                
                c_l = f_cl(new_attack)
                        
                c_d = f_cd(new_attack)
                        
                k.update_coefficients_cont(c_l,c_d,new_bank)
                        
                sim_status = k.evolve_system_2(integration_steps_per_learning_step,integration_step) 
            elif args.alg=='sarsa':
                A_t=greedy_action(Q[S_t])
                new_attack_angle, new_bank_angle=apply_action(S_t, A_t)
                sim_status=k.evolve_system(new_attack_angle, new_bank_angle, integration_steps_per_learning_step, integration_step)
                S_t1 = (new_attack_angle, new_bank_angle, k.beta())
                    
            
            
            if i<10: 
            
                r.append(k.position.r)
                
                theta.append(k.position.theta)
                
                phi.append(k.position.phi)
                
            if not sim_status ==0: 
            
                reward = -0.1
                
                done = True 
                
                new_state = state 
                
            else: 
                if args.alg=='td3':
                    new_state = np.asarray((new_attack, new_bank, k.beta()))
                           
                reward = k.reward(learning_step)
                
            if (count==int(horizon)-1 or k.fullyunrolled()): 
            
                done = True 
                
            score+=reward 
                    
            if args.alg=='td3':
                state = new_state 
            elif args.alg=='sarsa':
                S_t=S_t1
             
             
                    
            if i<10: 
                if args.alg=='td3':
                    alpha.append(state[0])
                    bank.append(state[1])
                    beta.append(state[2])
                elif args.alg=='sarsa':
                    alpha.append(pk.attack_angles[S_t[0]])
                    bank.append(pk.bank_angles[S_t[1]])
                    beta.append(S_t[2])
                power.append((reward/learning_step)*3600)   #KW 
                eff_wind_speed.append(k.effective_wind_speed())
                 
        cumulative_time.append(int_time)   
                
        rewards.append(score) 
        
        time_history.append(time) 
        
        KWh_per_second.append(score/time)
         
     

    r = np.array(r)
     
    theta = np.array(theta)
    
    phi = np.array(phi)
     
    alpha = np.array(alpha)
    
    bank = np.array(bank)
    
    beta = np.array(beta)
    
    x=np.multiply(r, np.multiply(np.sin(theta), np.cos(phi)))
    
    y=np.multiply(r, np.multiply(np.sin(theta), np.sin(phi)))
    
    z=np.multiply(r, np.cos(theta))


    np.save(args.path+"durations.npy", np.array(time_history))
    np.save(args.path+"cumulative_durations.npy", np.array(cumulative_time))
    np.save(args.path+"power.npy", np.array(power))
    np.save(args.path+"x.npy", x)
    np.save(args.path+"y.npy", y)
    np.save(args.path+"z.npy", z)
    np.save(args.path+"alpha.npy", alpha)
    np.save(args.path+"bank.npy", bank)
    np.save(args.path+"beta.npy", beta)

    mean_performance = os.path.join(args.path,"mean_performance.txt")
            
    with open(mean_performance,'w') as f: 
                
        f.write("average reward = "+str(np.mean(np.asarray(rewards)))+"\n")
        f.write("average time = "+str(np.mean(np.asarray(time_history)))+"\n")
        f.write("average power =" +str(np.mean(np.asarray(KWh_per_second)))+"\n")

    
    
               
                        
                                                
     
                        
            
                
                    
            
                
                    
            
            
            
            
        
if __name__ == "__main__": 

    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--path", default="./results/const/")
    parser.add_argument("--alg", default="sarsa")
    parser.add_argument("--wind", default="const") #const, lin or turbo
    parser.add_argument('--step',type=float,default=0.1)
    parser.add_argument("--eval_episodes", type=int, default=1e1)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument('--range_actions',action='append',default = None)  

    args = parser.parse_args() 
    initial_position = pk.vect(np.pi/6, 0, 20)
    initial_velocity = pk.vect(0, 0, 0)
    k = pk.kite(initial_position, initial_velocity, args.wind,continuous=(args.alg=='td3'))
    
    
    
    test(k, args) 
            
            
        
                
