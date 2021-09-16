""" 
simulation.py create a highway gym environment with Open gym standard. 
Gym builds the environment for an agent to do various experiments. 
the highway environment built with help of SUMO (Simulated Urban Mobility) for visualization and Traci (Traffic Control Interface)    
Goals: "ego" doesn't crash or out of the road, and "emg" reach to the end ASAP 
"""
import os 
import sys
from numpy.lib import arraypad
from sumolib import checkBinary
import optparse
import traci
import matplotlib.pyplot as plt 
import numpy as np 
import traci.constants as tc
import simulaiton_params as s 
import copy
import random
import warnings
from visualization import Visual 
warnings.simplefilter('always', UserWarning)
 
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)                                                     # Set SUMO environment variable 
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

# First class provide all the .xml files that we need for simulation
class Road (object):
    def __init__(self):        
        self.generate_nodes()
        self.generate_edges()            
        self.generate_routefile()
        self.sumo_sttings()
        self.sumo_cfg()
    
    #Generate nodes: 
    @staticmethod
    def generate_nodes():
        with open("highway.node.xml", "w") as nodes: 
            print ("""<nodes>
            <node id="0" x="0" y="0.0" />
            <node id="1" x="2000" y="0.0" />
            </nodes>""", file=nodes) 
    
    #Generate edges:
    @staticmethod
    def generate_edges(): 
        with open("highway.edg.xml", "w") as edges: 
            print("""<edges>
                    <edge from="0" id="add" to="1" numLanes="3" width="3.2" speed="60" />
                    </edges> """, file=edges)
    
    # Generate Route File:
    @staticmethod 
    def generate_routefile():
        with open ("highway.rou.xml", "w" ) as routes: 
            print (""" <routes>
            <vType id="emg" vClass="passenger" length="16.0" width="2.55" maxSpeed="50" guiShape="emergency" color="1,0,0" speedFactor ="1.5" />
            <vType id="ego" vClass="passenger" length="4.8" width="1.8" maxSpeed="50" guiShape="bus" color="0,1,0" lane ="1"  />
            <vType id="car" vClass="passenger" length="4.8" width="1.8" maxSpeed="30" guiShape="passenger/van" color="0,1,1"   />
            <route id="route0" edges="add"/>
            </routes>""", file=routes)
    
    #Sumo Settings 
    @staticmethod
    def sumo_sttings():
        with open ("highway.settings.xml", "w") as sumo_sttings: 
            print("""<viewsettings>
            <viewport x="750" y="0" zoom="600"/>
            <delay value="200"/>
            <scheme name="real world"/>
            <scheme>
            <polys polyName_show="1" polyName_color="0,0,0" polyName_size="100.00"/>
            </scheme>
            </viewsettings>""", file=sumo_sttings)
    
    #Sumo Config file
    @staticmethod 
    def sumo_cfg(): 
        with open ("highway.sumocfg", "w") as sumo_cfg:
            print("""<configuration>
                <input><net-file value="highway.net.xml"/>
                <route-files value="highway.rou.xml"/>
                <gui-settings-file value="highway.settings.xml"/>
                </input>
                <time><begin value="0"/><end value="1e15"/>
                </time>
                <processing><lanechange.duration value="0"/><lanechange.overtake-right value="true"/><collision.action value="warn"/>
                </processing>
                </configuration>""", file=sumo_cfg)    
        net_convert = "netconvert --node-files=highway.node.xml --edge-files=highway.edg.xml --output-file=highway.net.xml"
        os.system(net_convert)
        
# Second class provide all the vehicles. 
class Vehicles(object): 
    random.seed(64)
    np.random.seed(64)
    def __init__(self, sim_params): 
        self.nb_vehicles = sim_params["nb_vehicles"]
        self.box =  sim_params["box"]
        self.max_speed  = sim_params["max_speed_ego"]
        self.departSpeed = sim_params["departSpeed"]
        self.ego_depart_pos = sim_params["ego_depart_pos"]
        self.sumo_ctrl = sim_params['sumo_ctrl']

    def remove(self):
        for veh in traci.vehicle.getIDList(): 
            traci.vehicle.remove(veh)
        for i in range (2): 
            traci.simulationStep()

    def add(self): 
        for i in range (self.nb_vehicles):
            if i == 0: 
                veh_id ="ego"
                lane = np.random.randint(0,2)
                traci.vehicle.add(veh_id, 'route0', typeID="ego", departLane=lane, depart= None, 
                departPos= self.ego_depart_pos, departSpeed=self.departSpeed )              
            elif i==1:
                veh_id = "emg"
                lane = 1 
                traci.vehicle.add(veh_id, 'route0', typeID='emg', departLane=lane, depart= None,
                 departPos='base', departSpeed= self.departSpeed)
            else :
                veh_id = 'veh' + str(i).zfill(int(np.ceil(np.log10(i))))
                lane = np.random.choice([0,2])
                traci.vehicle.add(veh_id, "route0" , typeID="car", departLane=lane, arrivalLane=lane, depart=None,
                departPos= np.random.randint(100,1500))
        traci.simulationStep()                           
        self.vehicles = traci.vehicle.getIDList()    
    
    def control(self): 
        if self.sumo_ctrl is False: 
            # trun off all speed and lane controllert for the "ego"
            traci.vehicle.setSpeedMode("ego", 0)
            traci.vehicle.setLaneChangeMode("ego",0)
        else : 
            traci.vehicle.setSpeedMode("ego", -1)
        # Control of the speed "veh" type by SUMO 
        for i, veh in enumerate( self.vehicles [2:]):    
            traci.vehicle.setSpeed(veh, -1)
            traci.vehicle.setLaneChangeMode(veh, np.random.choice([0,2]))
        # Make sure "emg" type stay on the lane = 1 and with maximum speed        
        traci.vehicle.setSpeed("emg", self.max_speed)
        traci.vehicle.setLaneChangeMode("emg", 1)
        return self.vehicles

    def subs (self): 
        # Subsciption 
        traci.vehicle.subscribeContext("ego", tc.CMD_GET_VEHICLE_VARIABLE, self.box,
        [tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_SPEED_LAT, tc.VAR_ACCELERATION , tc.VAR_TYPE])
        filt_out = traci.vehicle.getContextSubscriptionResults("ego")
        return filt_out      

# Third class provide all we need for simulatin.

class Simulation(object):

    def __init__(self, sim_params):
        self.vehicles_sens  = 10
        self.positions = np.zeros([self.vehicles_sens,2])     # long and lat positions 
        self.speeds =  np.zeros([self.vehicles_sens, 1])      # long and lat speeds
        self.acccs =  np.zeros([self.vehicles_sens, 1])       # accs 
        self.action  = sim_params['action']
        self.min_road_speed =  sim_params["min_road_speed"]
        self.max_road_speed  = sim_params['max_road_speed']
        self.number_of_games = sim_params["number_of_games"]
        self.gui_use  = sim_params['gui_use']
        self.sumo_ctrl = sim_params['sumo_ctrl']
        self.col_penalty = sim_params["collision_penalty"] 
        self.out_of_road_penalty = sim_params['out_of_road_penalty']
        self.ego_emg_same_lane_penalty = sim_params['ego_emg_same_lane_penalty'] 
        self.change_lane_penalty = sim_params["lane_change_penalty"]
        self.speed_violation_penalty = sim_params["speed_penalty"]
        self.box  = sim_params["box"]
        self.length_road = 2000
        self.veh = Vehicles(sim_params = s.sim_params)
        self.viz = Visual(sim_params = s.sim_params)
        self.state = []

    def observation(self):
        subscriptions = self.veh.subs()
        if subscriptions is None: 
            return None 
        else: 
            for i, veh in enumerate(subscriptions):
                self.positions[i, 0] = np.array(subscriptions[veh][tc.VAR_POSITION])[0]/self.length_road
                self.positions[i, 1] = np.array(subscriptions[veh][tc.VAR_POSITION])[1]                                  
                self.speeds[i] = (np.array(subscriptions[veh][tc.VAR_SPEED]) - self.min_road_speed)/(self.max_road_speed - self.min_road_speed)
                self.acccs[i] = np.array(subscriptions[veh][tc.VAR_ACCELERATION])        # return accelaration  
                # 1: "ego" sense "emg" in filter box 2: "ego" not sense "emg" in filter box                 
                if subscriptions[veh][tc.VAR_TYPE] == 'ego' or subscriptions[veh][tc.VAR_TYPE] == 'veh':  
                    self.veh_t = 0 
                elif subscriptions[veh][tc.VAR_TYPE] == 'emg':
                    self.veh_t = 1
                self.veh_t = np.array(self.veh_t)
                # Get ride of the crazy position output of the SUMO net convert. 
                if self.positions[i, 1] == -8: 
                    self.positions[i, 1] = -1/2
                elif self.positions[i, 1] == -4.8: 
                    self.positions[i, 1] = 0
                elif self.positions[i, 1] == -1.6: 
                    self.positions[i, 1] = 1/2        
                # Flat and change the dtype of all
                positions = self.positions.flatten()
                speeds = self.speeds.flatten()
                self.state = np.concatenate((np.array([self.veh_t]), positions, speeds)).astype(np.float32)
        return self.state

    def action_func(self, rand_action): 
        # If SUMO_ctrl is off ego can do one of the following actions. 
        if self.sumo_ctrl is False:
        
            if rand_action == 0 :                      # no change lane no change speed 
                print ("action0: No change ")
                traci.vehicle.changeLaneRelative("ego", self.action[0][0], 1)
                traci.vehicle.setSpeed("ego", traci.vehicle.getSpeed("ego") + self.action[0][1])
                
            elif rand_action == 1 :                    # change lane to the right lane 
                print ("action1: change to the right lane ")
                traci.vehicle.changeLaneRelative("ego", self.action[1][0], 1)
                traci.vehicle.setSpeed("ego", traci.vehicle.getSpeed("ego") + self.action[1][1])
        
            elif rand_action == 2:                     # change lane to the left lane 
                print("aciton2: change to the left lane")
                traci.vehicle.changeLaneRelative("ego", self.action[2][0], 1 )
                traci.vehicle.setSpeed("ego", traci.vehicle.getSpeed("ego") + self.action[2][1])

            elif rand_action == 3 :                   # 5 m/s increase speed 
                print ("action3: speed up ")
                traci.vehicle.changeLaneRelative("ego", self.action[3][0], 1)
                traci.vehicle.setSpeed("ego", traci.vehicle.getSpeed("ego") + self.action[3][1])

            elif rand_action == 4:                    # 5 m/s decrease speed 
                print ("action 4: speed down ")
                traci.vehicle.changeLaneRelative("ego", self.action[4][0], 1)
                traci.vehicle.setSpeed("ego", self.speeds[0,0] + self.action[4][1])
                        
    def step(self, rand_action):
        # What will happen to the agent after taking an action. 
        self.ego_collision = False
        self.out_of_road =  False
        self.ego_reach_end = False 
        self.emg_reach_end =  False 
        self.out_of_access = False 
        self.ego_emg_same_lane = False
        self.change_lane = False
        self.speed_violation = False 
        done  =  False  
        right_shoulder = False 
        left_shoulder =  False  
        
        collision =  traci.simulation.getCollidingVehiclesNumber() > 0

        if collision is True :
            print ("NONO") 
            self.ego_collision = True 
            
        if self.sumo_ctrl is False:
            if traci.vehicle.getPosition("ego")[1] == -1.6 : 
                if rand_action == 2:
                    left_shoulder = True 
                    self.state[2] = -1 
            elif traci.vehicle.getPosition("ego")[1] == -8 :
                if rand_action == 1:
                    right_shoulder = True
                    self.state[2] = 1 
        
        if left_shoulder ==True or right_shoulder==True : 
            self.out_of_road = True
            self.ego_collision = False
            print ("OUTOFROAD")
                
        if self.state[0] == 1:                                                                              # ego sense emg in range 
            if traci.vehicle.getPosition("emg")[0] - traci.vehicle.getPosition("ego")[0] > self.box  :      # emg pass the ego 
                print ('out of access ')
                self.out_of_access = True 
                
        if traci.vehicle.getPosition("ego")[0] > 1000: 
            print ("ego reach end ")
            self.ego_reach_end = True 
            
        if traci.vehicle.getPosition("emg")[0] > 1000: 
            print ("emg reach end ")
            self.emg_reach_end = True 
            
        if self.ego_collision: 
            done  = True 
        elif self.out_of_road: 
            done =  True 
        elif self.out_of_access:                    # 5 reasons of termination! 
            done = True 
        elif self.emg_reach_end: 
            done =  True 
        elif self.ego_reach_end: 
            done = True 
        
        if np.abs(self.action[rand_action][0]): 
            self.change_lane = True 
            print ("Lane Changed")

        if self.state[0] == 1:                      # Ego sense emg in range
            if traci.vehicle.getPosition("ego")[1] == traci.vehicle.getPosition("emg")[1]: 
                print ("ego_emg_same_lane")
                self.ego_emg_same_lane  = True

        if self.state[0] == 1 : 
            speed_ego = traci.vehicle.getSpeed('ego')
            speed_emg  = traci.vehicle.getSpeed('emg')
            if speed_ego > speed_emg * 0.95 : 
                self.speed_violation = True 
                print ("speed_violation")  # Can Change by if speed_t1("ego") > speed_t0 ("ego")

        info = []
        info.append (self.ego_collision)
        info.append(self.out_of_road)
        info.append(self.emg_reach_end)
        info.append(self.ego_emg_same_lane)
        info.append(self.speed_violation)
        
        
        reward = self.reward_cal(ego_collision=self.ego_collision, out_of_road = self.out_of_road,
        change_lane= self.change_lane, ego_emg_same_lane= self.ego_emg_same_lane, speed_violation = self.speed_violation)
        
        return self.state, reward, done, info     

    def reward_cal (self, ego_collision, out_of_road, change_lane, ego_emg_same_lane, speed_violation):
        self.reward = 0 

        if ego_collision: 
            self.reward -= self.col_penalty

        if out_of_road: 
            self.reward -= self.out_of_road_penalty

        if change_lane: 
            self.reward -= self.change_lane_penalty 

        if ego_emg_same_lane: 
            self.reward -= self.ego_emg_same_lane_penalty
            
        if speed_violation: 
            self.reward -= self.speed_violation_penalty

        return self.reward

    def run (self):
        options = get_options() 
        options.nogui  = self.gui_use         # Off or on simulation 
        if options.nogui:
            sumoBinary = checkBinary('sumo-gui' )
        else:
            sumoBinary = checkBinary('sumo')
        
        steps = []
        rewards = []
        ego_emg_same_list = []
        out_of_roads_list = []
        change_lane_list = []
        accidents_list = []
        speed_violation_list  = []
    
        for i in range (self.number_of_games):
            np.random.seed(i)
            traci.start([sumoBinary, "-c", "highway.sumocfg", "--start"]) 
            veh = Vehicles(s.sim_params)
            veh.remove()
            done = False  
            accident_times = 0 
            out_road_times = 0 
            change_lane_times = 0 
            total_rewards = 0 
            speed_violation_times = 0 
            same_lane_times = 0 
            reward = 0
            step = 0 
            for i in range (2):
                traci.simulationStep()
            veh.add()
            veh.control()

            while done is False : 
                state_t1 = sim_1.observation()
                print ("s1:", state_t1)    
                rand_action  = random.randrange(len(self.action))
                sim_1.action_func(rand_action)
                traci.simulationStep()
                state_t2 = sim_1.observation()
                state_t2, reward, done, info = sim_1.step(rand_action)
                print("s2:", state_t2)
                print ("done", done)
                print ("episode_reward", reward)
                step += 1
                total_rewards += reward 
                print ("total_rewards", total_rewards)

                if self.ego_emg_same_lane: 
                    same_lane_times += 1

                if self.change_lane: 
                    change_lane_times += 1 
                
                if self.speed_violation: 
                    speed_violation_times += 1 

            change_lane_list.append(change_lane_times)
            ego_emg_same_list.append(same_lane_times)                
            speed_violation_list.append(speed_violation_times)
            steps.append(step)
            rewards.append(total_rewards)           

            traci.close() 

            if self.ego_collision: 
                accident_times = 1 
            accidents_list.append(accident_times)
    
            if self.out_of_road: 
                out_road_times = 1
            out_of_roads_list.append(out_road_times)

        print ("out_of_road_list", out_of_roads_list)
        print ('collision_list', accidents_list)
        print ("same_lane_with_emg_list", ego_emg_same_list)
        print ("rewards", rewards)
        print ("step", steps)
        print ("change_lane_list", change_lane_list)
        print ("speed_violation_list", speed_violation_list)

        # Plot additional features 
        self.viz.general_plot (accidents=accidents_list, out_of_roads =out_of_roads_list,
        same_lane_emg=ego_emg_same_list, change_lane_times=change_lane_list, steps= steps, speed_violation=speed_violation_list)
        # Plot reward function 
        self.viz.reward_plot(reward= rewards)

        return (state_t1, rand_action, state_t2, reward, done)        

if __name__ == "__main__": 
    road = Road()
    sim_1  =  Simulation(sim_params=s.sim_params)
    sim_1.run()
############################################# End of Simulation ##################################