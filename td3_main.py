
import numpy as np 
import argparse 
import os 
import learning.pykite as pk 
from utility import *
import pandas as pd 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt 
#from classes import *
from classes_ import*
from test import test 

ACTION_NOISE = 0.2
ATTACK_INF_LIM = -5 
ATTACK_SUP_LIM = 18 
BANK_INF_LIM = -3 
BANK_SUP_LIM=3






def td3(args): 

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
    
    EPISODES = int(args.episodes) 
    
    c_lr = args.critic_lr
    
    a_lr = args.actor_lr
    
    step= args.step 
    
    integration_step = 0.001 
    
    duration = 300 
    
    episode_duration= int(duration) 
    
    horizon = int(episode_duration/step) 
    
    integration_steps_per_learning_step=int(step/integration_step)
    
    dir_name = os.path.join(args.path,"data")
    
    dir_nets = os.path.join(args.path,"nets")
    
    if not os.path.exists(dir_name):
    
        os.makedirs(dir_name)
        
    best_score = -200 
    
    score_history = [] 
    
    data_name= os.path.join(dir_name,"data_training.txt") 
               
    pict_name = os.path.join(dir_name,"average_reward.png") 
    
    agent =Agent(3,2,critic_lr=c_lr,actor_lr=a_lr,gamma=0.99,chkpt_dir=dir_nets)
    
    agent.manual_initialization()
    
    for i in range(EPISODES): 
    
        done = False 
        
        score = 0 
        
        k.reset(initial_position, initial_velocity, args.wind) 
        
        initial_beta = k.beta(continuous=True) 
        
        S_t=(np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta)
        
        c_l = f_cl(S_t[0])
        
        c_d = f_cd(S_t[0])
        
        k.update_coefficients_cont(c_l,c_d,S_t[1])     
        
        state = np.asarray(S_t) 
        
        count = 0
        
        while not done: 
        
            count +=1 
            
            action = agent.choose_action(state,ACTION_NOISE)
            
            new_attack, new_bank = state[0]+action[0]*range_actions[0], state[1]+action[1]*range_actions[1]
            
            new_attack = np.clip(new_attack,ATTACK_INF_LIM,ATTACK_SUP_LIM) 
            
            new_bank = np.clip(new_bank,BANK_INF_LIM,BANK_SUP_LIM)
            
            c_l = f_cl(new_attack)
            
            c_d = f_cd(new_attack)
            
            k.update_coefficients_cont(c_l,c_d,new_bank)
            
            sim_status = k.evolve_system_2(integration_steps_per_learning_step,integration_step)
            
            if not sim_status == 0: 
            
                reward = -0.1 
                
                done = True 
                
                new_state = state 
                
            else: 
            
                new_state = np.asarray((new_attack, new_bank, k.beta(continuous=True)))
                
                reward = k.reward(step) 
                
            if (count==int(horizon) -1 or k.fullyunrolled()): 
            
                done = True 
                
            agent.store_transition(state, action, reward, new_state, done) 
            
            agent.train() 
            
            score += reward 
            
            state = new_state 
            
        score_history.append(score) 
        
        avg_score = np.mean(score_history[-20:])
        
        if avg_score > best_score: 
                   
            best_score = avg_score 
                       
            agent.save_models() 
            
            
    
    x = [i+1 for i in range(EPISODES)]   
    
    plot_average_reward(x,score_history,pict_name) 
    
    with open(data_name,'w') as f: 
    
        for i in range(0,len(score_history)): 
        
            f.write(str(score_history[i])+"\n") 
            
            
    
         
    
    test(args)           
            
    
        
    
    

    
    
