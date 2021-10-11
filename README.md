# Autonomous-Driving
# Self-driving car and Emergency vehicle. 
Autonomous Vehicles (AVs) are the future of road transportation which can bring us safety, efficiency, and productivity. But staying in the lane is not enough anymore and AVs need to train for doing more complicated edge cases. In this project, I try to solve one of these edge cases with help of Deep Reinforcement Learning(DRL).

Why I am thinking about this project? 
Two reasons, First, I am crazy about all cars either the 1980 Toyota Corolla or Tesla model S. Second, I watch the AlphaGo movie https://www.youtube.com/watch?v=WXuK6gekU1Y so need to have a project about Reinforcement Learning!. 

OK!. what is Deep Reinforcement Learning? 

**Agent**, do **Actions**, in **Environment**, and **Reward** determine the agent's performance. Train model as an agent brain, more training, again training, save trained model to new model, train, train, train. Agent turn to  a **hero**. That is my abstract definition of the (DRL) but you can find great materials for learning more about (DRL) in the following. 

--> Hollo World of DRL in Human-level control through deep reinforcement learning article https://www.nature.com/articles/nature14236

--> Reinforcement Learning - Goal Oriented Intelligence here https://deeplizard.com/learn/playlist/PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv. 

Let's start to define the main components in this project:

**Agent**: Autonomous vehicle (ego) which learns how to drive normally in a 3 lane highway and also change speed and lane when approached by an emergency vehicle.

**Environment**: As you guess 3 lane highway but needs to be gym-like! what is a  gym? take a look at https://gym.openai.com/. There are some open-source simulators for building our highway gym or generally for having a self-driving environment. CARLA https://carla.org/ and Simulated Urban Mobility (SUMO) https://www.eclipse.org/sumo/. In this project I choice SUMO for creating a gym environment, why? 2D is enough for this project, more computational friendly compare to CARLA, Traffic Control Interface (Traci) as a server-client can make a good connection between agent and environment. The other option is to develop our own gym as a great example is https://github.com/eleurent/highway-env but I prefer to trust the people in the German Aerospace Center and build the environment on the top of their platform and work with Traci as a connection of all the python code to the simulation. There is another platform that developed UC Berkeley Mobile Sensing Lab https://flow-project.github.io/ and it makes a connection between SUMO and other components of (DRL). 
For how to install SUMO and Hello world of SUMO you can take a look on this SUMO documentation https://sumo.dlr.de/docs/Tutorials/Hello_World.html. 
At the end, we need five .xml files and one sumo config file .sumocfg file. All these files creat by Road as a first class in simulation.py.  

**Actions**: I consider 5 actions for the ego that can do in each step of one episode.
            a0 : no change speed, no change lane
            a1 : change to the left lane 
            a2 : change to the right lane 
            a3 : increase speed 
            a4:  decrease speed 

**Reward** : I want the agent to learn both normal driving and making the best decision when approached by the emergency vehicle. We consider some penalties and the agent needs to minimize the sum of these penalties at each step of one episode. The following equation will describe the penalty at each step of one episode. 

.py description: 
    Simulation.py: generate nodes, edges, route, settings, configuration file, remove vehicles, add vehicles, control vehicle, subscribe to the objects, collect states, do actions, next observation, calculate the reward. 

    simulation_params.py all the parameters. 

    visualizaiton.py some simple plots for analyzing the results. 
    
    DQNAgent.py Create replay memory, create model, get Q-values, train the model 

    Training.py connects all the essentials for starting the simulation and training. 