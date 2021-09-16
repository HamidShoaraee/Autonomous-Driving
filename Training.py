"""
In this file we start simulation and try to train the agent to take the best actions. 
This file connect modules of the both simulation.py and DQNAgent.py
"""
import numpy as np
from numpy.core import numeric 
from simulation import Vehicles, Road, Simulation
import simulaiton_params as s 
from DQNAgent import Memory, Agent, EpsilonGready
import traci
import random 
from sumolib import checkBinary
import tensorflow as tf 
import matplotlib.pyplot as plt
from visualization import Visual 

####################################### Run simulation and Training ###############################
number_of_games = s.sim_params["number_of_games"]
gui_use  = s.sim_params['gui_use']                    # Train with gui not a good idea ! 
sumo_ctrl = False                                     # Training without help of SUMO                      
action = s.sim_params['action']                       

epsilon = s.sim_params['epsilon']
batch_size = s.sim_params['batch_size']
min_size_memory = s.sim_params['min_size_memory'] 
max_size_memory = s.sim_params['max_size_memory'] 

veh = Vehicles(sim_params= s.sim_params)
sim_1  = Simulation(sim_params=s.sim_params)
replay_memory = Memory(capacity=50)
agent = Agent (sim_params = s.sim_params)
epsilon_gredy = EpsilonGready(sim_params = s.sim_params)
visual = Visual(sim_params=s.sim_params)

np.random.seed(32)
experiences = []
rewards = []
reward_avg_list = []
steps = []
ego_emg_same_list = []
out_of_roads_list = []
emg_reach_end_list = []
accidents_list = []
speed_violation_list  = []

for i in range (number_of_games): 
    sumoBinary = checkBinary('sumo')     # Traing + gui --> computational cost! 
    traci.start([sumoBinary, "-c", "highway.sumocfg", "--start"])    # Start the simulation. 
    veh.remove()
    done  =  False
    step = 0
    episode_reward = 0 
    accident_times = 0 
    out_road_times = 0 
    emg_reach_end = 0 
    total_rewards = 0 
    speed_violation_times = 0 
    same_lane_times = 0 
    initial_eps = 1 
    eps_decay = 0.0005
    eps = initial_eps - (eps_decay * i)
    for i in range (2):                 # Two simulation steps at the begining before adding vehicles 
        traci.simulationStep()
    veh.add()
    veh.control()

    while done is False : 
        current_state = sim_1.observation()         
        # exploration v.s exploitation (First more explore then explot)
        if eps > epsilon :           
            rand_action = random.randrange(len(action))                        # First explore 
        else : 
            rand_action = np.argmax (agent.get_qs(state = current_state))      # Then explot (predict the next state)
        sim_1.action_func(rand_action)
        traci.simulationStep()
        new_state = sim_1.observation()
        # info: ego_collision, out_of_road, change_lane, ego_emg_same_lane, speed_violation
        new_state, reward,  done, info  = sim_1.step(rand_action) 
        
        if info[2] == True : 
            emg_reach_end += 1 
        if info[3] == True : 
            same_lane_times += 1 
        if info[4] == True : 
            speed_violation_times += 1 
        
        step += 1 
        episode_reward += reward
        sample = [current_state, rand_action, new_state, reward, done]
        experiences.append(sample)

    emg_reach_end_list.append(emg_reach_end)  
    ego_emg_same_list.append(same_lane_times)
    speed_violation_list.append(speed_violation_times)
    steps.append(step)

    traci.close()
    
    if info [0] == True : 
        accident_times = 1 
    accidents_list.append(accident_times)

    if info[1] == True : 
        out_road_times = 1 
    out_of_roads_list.append(out_road_times)

    if len(experiences) < max_size_memory:        
        experiences.append(sample)

    if len(experiences) == max_size_memory: 
        experiences.pop(0)
        
    # Get a batch form filled memory.
    if len (experiences) > batch_size * 2  : 
        minibatch = random.sample(experiences, batch_size)
        experiences = [] 
        agent.train( termination_state = done , minibatch = minibatch)
        current_state = new_state
    
    rewards.append(episode_reward)

s = np.arange(0, number_of_games + 1, step=5)
len_s = len(s)
for i in range (len_s - 1): 
    reward_avg = np.average(rewards[s[i]:s[i+1]])
    reward_avg_list.append(reward_avg)

print ("rewards",rewards)
print ("reward_length", len(rewards))
print ("average_rewards", reward_avg_list)
print ("avg_reward_len", len(reward_avg_list))

# Save the model
agent.save_model()
# visual.reward_plot(reward= rewards)
visual.general_plot (accidents=accidents_list, out_of_roads =out_of_roads_list,
same_lane_emg=ego_emg_same_list, emg_reach_end=emg_reach_end_list, steps= steps, speed_violation=speed_violation_list)
#Scatter Plot reward function
visual.reward_plot(reward = rewards)
#Scatter Plot average reward function 
visual.reward_avg_plot(avg_reward = reward_avg_list)
#Line Plot 
visual.reward_plot_plot(reward=rewards)  

visual.totext(rewards=rewards, avg_reward= reward_avg_list, accidents=accidents_list, out_of_roads=out_of_roads_list, same_lane_emg= ego_emg_same_list, 
                emg_reach_end=emg_reach_end_list, steps=steps, speed_violation= speed_violation_list)



