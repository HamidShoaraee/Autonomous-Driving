import os 
import numpy as np 
##########################################Genarl simulation parameters###########################
sim_params  = {}
sim_params['sumo_ctrl'] = False         
sim_params['gui_use']   = False      
sim_params['action']  = [[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]]
sim_params["number_of_games"] = 5000
sim_params["box"] =  100 
sim_params["nb_vehicles"] = 18                     
sim_params["departSpeed"]  = 15 
sim_params["max_speed_emg"] = 50
sim_params['max_speed_veh'] =  30                             
sim_params["max_speed_ego"] = 30  
sim_params["min_road_speed"] =  10     # m/s                        
sim_params["max_road_speed"] =  50     # m/s
sim_params["ego_depart_pos"] = np.random.randint(20, 200)       
sim_params["emg_depart_pos"] = "base"                             # Departure pos of the emg 
sim_params["veh_depart_pos"] =  np.random.randint(0, 500)

######################################### Training_Prams ######################################## 
sim_params['input_shape'] = (31,)                                 # Need to make sure 
sim_params['output_shape'] = (np.shape(sim_params['action'])[0])
sim_params['max_size_memory'] = 100
sim_params['min_size_memory'] = 0
sim_params["num_layers"] = 3 
sim_params["num_nodes"] = 25 
sim_params["batch_size"] = 64
sim_params["learning_rate"]  = 0.0001
sim_params['discount'] = 0.99
sim_params["update_target_every"] = 5
sim_params['start_epsilon']  = 1
sim_params['end_epsilon'] = 0
sim_params['epsilon_decay']  = 0.001                      # This can tune based on the number of games 
sim_params['epsilon'] = 0.57  
sim_params['number_of_training'] = 2000

sim_params["collision_penalty"] = 150
sim_params['ego_emg_same_lane_penalty'] = 5
sim_params['out_of_road_penalty'] = 200      
sim_params["speed_penalty"] = 1                      
sim_params["lane_change_penalty"] = 1