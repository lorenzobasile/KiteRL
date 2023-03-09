import os 
import argparse
import numpy as np
import learning.pykite as pk 
from utility import *
import pandas as pd 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt 
from classes_ import *

ACTION_NOISE = 0.2
ATTACK_INF_LIM = -5 
ATTACK_SUP_LIM = 18 
BANK_INF_LIM = -3 
BANK_SUP_LIM=3


def test(args): 

    if(args.range_actions) is not None: 
    
        if(len(args.range_actions[0])>1): 
        
            range_actions = np.array([args.range_actions[0][0],args.range_actions[0][2]])
            range_actions = range_actions.astype(np.float64)
            
        else: 
        
            range_actions=np.array([args.range_actions[0][0],args.range_actions[0][0]])
            range_actions = range_actions.astype(np.float64)
            
    else: 
    
        range_actions = np.array([1,1]) 
        
        
    Cl_angles = read_file('CL_angle.txt') 
    
    Cl_values = read_file('CL_value.txt') 
    
    Cd_angles = read_file('CD_angle.txt') 
    
    Cd_values = read_file('CD_value.txt') 
    
    Cl_data = pd.DataFrame({'cl_angles':Cl_angles,'cl_values':Cl_values}) 
    
    Cd_data = pd.DataFrame({'cd_angles':Cd_angles,'cd_values':Cd_values}) 
    
    f_cl  = interpolate.interp1d(Cl_data.cl_angles, Cl_data.cl_values, kind='linear')
    
    f_cd  = interpolate.interp1d(Cd_data.cd_angles, Cd_data.cd_values, kind='linear')
    
    initial_position = pk.vect(np.pi/6, 0, 20)
    
    initial_velocity = pk.vect(0, 0, 0)
    
    k = pk.kite(initial_position, initial_velocity,args.wind)
    
    if(args.eval_episodes<25): 
    
        EPISODES=25
        
    else : 
    
        EPISODES = args.eval_episodes
        
    step= args.step 
    
    integration_step = 0.001
    
    duration = 300 
    
    episode_duration= int(duration) 
    
    horizon = int(episode_duration/step) 
    
    integration_steps_per_learning_step=int(step/integration_step)
    
    dir_name_data = os.path.join(args.path,"test")
    
    dir_nets = os.path.join(args.path,"nets")
    
    if not os.path.exists(dir_name_data):
    
        os.makedirs(dir_name_data)
        
    file_name = os.path.join(dir_name_data,"average_reward") 
    
    score_history = [] 
                
    time_history = [] 
                
    KWh_per_second = []
    
    r = []
    
    theta = [] 
    
    phi = [] 

    alpha = [] 
    
    bank = [] 

    beta = []
    
    power = []
    
    time_2 = []
    
    int_time = 0
                
    agent =Agent(3,2,chkpt_dir=dir_nets)#, max_action =range_actions)
            
    agent.load_actor() 
    
    eff_wind_speed = []
    
    for i in range(EPISODES): 
    
        done = False 
        
        time = 0 
        
        score = 0 
        
        k.reset(initial_position, initial_velocity,args.wind)
        
        initial_beta = k.beta(continuous=True) 
                
        S_t=(np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta)
                
        c_l = f_cl(S_t[0])
        
        c_d = f_cd(S_t[0])
        
        k.update_coefficients_cont(c_l,c_d,S_t[1])
        
        state = np.asarray(S_t) 
            
        count = 0 
                
        while(not done): 
        
            time+=step
                    
            int_time += 1 
                    
            action = agent.choose_action(state,0,test=True)
                    
            new_attack, new_bank = state[0]+action[0]*range_actions[0],state[1]+action[1]*range_actions[1]
            
            new_attack = np.clip(new_attack,ATTACK_INF_LIM,ATTACK_SUP_LIM) 
            
            new_bank = np.clip(new_bank,BANK_INF_LIM,BANK_SUP_LIM)
            
            c_l = f_cl(new_attack)
                    
            c_d = f_cd(new_attack)
                    
            k.update_coefficients_cont(c_l,c_d,new_bank)
                    
            sim_status = k.evolve_system_2(integration_steps_per_learning_step,integration_step) 
            
            if i<10: 
            
                r.append(k.position.r)
                
                theta.append(k.position.theta)
                
                phi.append(k.position.phi)
                
            if not sim_status ==0: 
            
                reward = -0.1
                
                done = True 
                
                new_state = state 
                
            else: 
            
                new_state = np.asarray((new_attack, new_bank, k.beta(continuous=True)))
                        
                reward = k.reward(step)
                
            if (count==int(horizon) -1 or k.fullyunrolled()): 
            
                done = True 
                
            score+=reward 
                    
            state = new_state 
             
             
                    
            if i<10: 
             
                alpha.append(state[0])
                bank.append(state[1])
                beta.append(state[2])
                power.append((reward/step)*3600)   #KW 
                eff_wind_speed.append(k.effective_wind_speed())
                 
        time_2.append(int_time)   
                
        score_history.append(score) 
        
        time_history.append(time) 
        
        KWh_per_second.append(score/time)
         
     
    x = [i for i in range(EPISODES)]
            
    mean_score = np.asarray(score_history) 
                
    mean_score= np.mean(mean_score) 
                
    mean_time = np.asarray(time_history) 
                
    mean_time = np.mean(mean_time) 
                
    mean_power = np.asarray(KWh_per_second)
                
    mean_power = np.mean(mean_power)    
    
    power_1 = power[:time_2[0]]
    
    power_2 = power[time_2[0]:time_2[1]]
    
    power_3 = power[time_2[1]:time_2[2]]
    
    power_4= power[time_2[2]:time_2[3]]
    
    power_1 = runn_average(power_1,5)
    
    power_2 = runn_average(power_2,5)
    
    power_3 = runn_average(power_3,5)
    
    power_4 = runn_average(power_4,5)
    
    duration_1 = [step*i for i in range(len(power_1))]
    
    duration_2 = [step*i for i in range(len(power_2))]
    
    duration_3 = [step*i for i in range(len(power_3))]
    
    duration_4= [step*i for i in range(len(power_4))]
        
         
            
        
    plt.style.use(['seaborn-whitegrid','tableau-colorblind10'])
            
    file_name = os.path.join(dir_name_data,"state_space.png")
            
    alpha_tot = alpha[time_2[5]:time_2[8]]
            
    bank_tot = bank[time_2[5]:time_2[8]]
            
    beta_tot = beta[time_2[5]:time_2[8]]
    durat = [step*i for i in range(len(beta_tot))]
    plt.plot(durat,alpha_tot,linewidth=1.5)
    plt.plot(durat,bank_tot,linewidth=1.5)
    plt.xlabel("time (s)")
    plt.ylabel("Angle (deg)")

    plt.legend(['Attack angle', 'Bank Angle',], loc='upper right')
            
    plt.savefig(file_name)
            
    plt.close()
    
    file_name = os.path.join(dir_name_data,"power.png")
        
    plt.plot(duration_1,power_1,linewidth=1.5)
    
    plt.plot(duration_2,power_2,linewidth=1.5)
        
    plt.plot(duration_3,power_3,linewidth=1.5)
   
    
           
    
    plt.xlabel('time (s)') 
    
    plt.ylabel('Generated Power (kW)')
    
    plt.savefig(file_name)
    
    plt.close()
    file_name = os.path.join(dir_name_data,"1_Ep_power.png")
    plt.plot(duration_2,power_2,linewidth=1.5)
    plt.xlabel('time (s)') 
    
    plt.ylabel('Generated Power (kW)')
    
    plt.savefig(file_name)
    
    plt.close()
    file_name = os.path.join(dir_name_data,"Effective_wind_speed_3_ep.png")
    wind_1 = eff_wind_speed[:time_2[0]]
    wind_2 =eff_wind_speed[time_2[0]:time_2[1]]
    wind_3 = eff_wind_speed[time_2[1]:time_2[2]]
    plt.plot(duration_1,wind_1,linewidth=1.5)
    
    plt.plot(duration_2,wind_2,linewidth=1.5)
        
    plt.plot(duration_3,wind_3,linewidth=1.5)
    plt.xlabel('time (s)') 
    
    plt.ylabel('Effective wind speed norm (m/s)')
    
    plt.savefig(file_name)
    plt.close()
    file_name = os.path.join(dir_name_data,"Effective_wind_speed_1_ep.png")
            
    
    plt.plot(duration_2,wind_2,linewidth=1.5)
        
            
    plt.xlabel('time (s)') 
    
    plt.ylabel('Effective wind speed norm (m/s)')
        
        
    
    plt.savefig(file_name)
    plt.close()
            
    file_name = os.path.join(dir_name_data,"duration")
    
    times = runn_average(time_history,5)
    
           
        
    plt.plot(x,times,linewidth=1.5)
    
    plt.title("Time required to complete an episode")
    
    plt.xlabel("Episode")
     
    plt.ylabel("Time (s) ")
    
    plt.savefig(file_name)
    
    plt.close()
        
    scores = runn_average(score_history,10) 
            
    file_name = os.path.join(dir_name_data,"energy_production")
            
    plt.plot(x,scores,linewidth=1.5)
            
    #plt.title("Energy produced in different episodes")
            
    plt.xlabel("Episode") 
            
    plt.ylabel("KWh ")
            
    plt.savefig(file_name)
            
    plt.close()
            
    r = np.array(r)
     
    theta = np.array(theta)
    
    phi = np.array(phi)
     
    alpha = np.array(alpha)
    
    bank = np.array(bank)
    
    beta = np.array(beta)
    
    x=np.multiply(r, np.multiply(np.sin(theta), np.cos(phi)))
    
    y=np.multiply(r, np.multiply(np.sin(theta), np.sin(phi)))
    
    z=np.multiply(r, np.cos(theta)) 
    
    for i in range(4): 
    
        if(i==0): 
        
            xt= x[0:time_2[0]]   
            
            yt= y[0:time_2[0]]
            
            zt =z[0:time_2[0]] 
        
            name_ = "traj_0"
            
            file_name = os.path.join(dir_name_data,name_)
            
            _ =plot_trajectory(xt,yt,zt,file_name)
                    
        else: 
        
            xt= x[time_2[i-1]:time_2[i]]   
            
            yt= y[time_2[i-1]:time_2[i]] 
            
            zt =z[time_2[i-1]:time_2[i]]
            
            name_="traj_"+str(i)
            
            file_name = os.path.join(dir_name_data,name_)
            
            _ = plot_trajectory(xt,yt,zt,file_name)
                
                
    x_ = x[:time_2[2]]
    y_=y[:time_2[2]]
    z_=z[:time_2[2]]
            
    t_time = [step*i for i in range(len(x_))]
            
    file_name = os.path.join(dir_name_data,"coordinate_3_phases.png")
    plot_distance(x_,y_,z_,t_time,file_name)
    total_time = [step*i for i in range(len(r))]
    
    file_name = os.path.join(dir_name_data,"coordinate_movement.png")
    
    plot_distance(x,y,z,total_time,file_name)
    
    
    #print("step",step,"learning rate=",)
        
    print("Average reward = ",mean_score)
                
    print("Average time = ", mean_time) 
                
    print("Average power = ",mean_power, "KWh/s") 
            
    mean_performance = os.path.join(dir_name_data,"mean_performance")
            
    with open(mean_performance,'w') as f: 
                
        f.write("average reward = "+str(mean_score)+"\n")
        f.write("average time = "+str(mean_time)+"\n")
        f.write("average power =" +str(mean_power)+"\n")
               
                        
                                                
     
                        
            
                
                    
            
                
                    
            
            
            
            
        
if __name__ == "__main__": 

    parser = argparse.ArgumentParser() 
    
    parser.add_argument('--episodes',type = int,default=2000) 
    
    parser.add_argument('--wind_type',default="const") 
    
    parser.add_argument('--step',type=float,default=0.1)
    
    parser.add_argument('--critic_lr',type = float, default = 0.001) 
    
    parser.add_argument('--actor_lr',type = float, default = 0.001) 
    
    parser.add_argument('--save_dir',default =  "dati_training/") 
    
    parser.add_argument('--test_episodes', type = int, default=500)
     
    parser.add_argument('--range_actions',action='append',default = None) 
    
    
    
    args = parser.parse_args() 
    
    test(args) 
            
            
        
                
