"""
Implement DQN algorithm from paper https://www.nature.com/articles/nature14236?wm=book_wap_0005 
DQN is base on the Q-learning algorith as a tabular algorithm which has use the power of  DNN as a general approximator. 
Experiences store in the memory and feed to the DNN as an input and output layer calculate the q-value of each action. 
Based on the experiences and action we train a model that can make best decision with new experiences. 
1. Initialize replay memory capacity.
2. Initialize the policy network with random weights.
3. Clone the policy network, and call it the target network.
4. For each episode:
    Initialize the starting state.
    For each time step:
    Select an action.
    Via exploration or exploitation
    Execute selected action in an emulator.
    Observe reward and next state.
    Store experiences in replay memory.
    Sample random batch from replay memory.
    Preprocess states from batch.
    Pass batch of preprocessed states to policy network.
    Calculate loss between output Q-values and target Q-values.
        Requires a pass to the target network for the next state
    Gradient descent updates weights in the policy network to minimize loss.
        After time steps, weights in the target network are updated to the weights in the policy network.
"""
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras import losses
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_array_ops import shape
import simulaiton_params as s 
import random 
import numpy as np
import math 
from simulation import Simulation 
import os 

class Memory (object): 
    def __init__(self, capacity): 
        self.memory = []                                # Samples on the replay memory 
        self.capacity = capacity
        self.push_count = 0
        batch_size  = 10
        
    def push (self, experience): 
        if len (self.memory) < self.capacity:
            self.memory.append(experience)
        else : 
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1 
        
    def sample(self, batch_size):                   # Use this samples in Replay memory 
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size): 
        return len(self.memory) >= batch_size


class EpsilonGready(object): 
    def __init__(self, sim_params): 
        self.start_epsilon = sim_params['start_epsilon'] 
        self.end_epsilon  = sim_params['end_epsilon']
        self.decay = sim_params['epsilon_decay']

    def exploration_rate (self, current_step):
        return self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-1 *current_step *self.decay)

class Agent (object):
    def __init__(self, sim_params):
        self.input_shape = sim_params['input_shape']            # Shape of input layer
        self.output_shape = sim_params['output_shape']          # Shape of output layer 
        self.num_layers = sim_params['num_layers']              # Number of hidden layers 
        self.num_nodes =  sim_params['num_nodes']               # Nodes in each layer 
        self.batch_size = sim_params['batch_size']              # Batch size in replay memory 
        self.learning_rate = sim_params['learning_rate']        # Learning_rate constant 
        self.discount  = sim_params['discount']                 # Reward discount 
        self.min_size_memory = sim_params['min_size_memory']
        self.Update_target_every  = sim_params["update_target_every"]
        self.number_of_training = sim_params['number_of_games']
        self.target_update_counter = 0 
        self.replay_memory = Memory(capacity=100)
        self.epsilon_greedy = EpsilonGready(sim_params=s.sim_params)
        self.model = self.create_model()                        # Policy network 
        self.target_model = self.create_model()                 # Target network 
        self.target_model.set_weights(self.model.get_weights())
        
    def create_model(self):
        """
        for now we consider a feedforward fully connected netwok. 
        """
        model =  keras.models.Sequential()
        input_1 =  keras.layers.Input(shape =(31,))
        hidden_1 = keras.layers.Dense(128, activation='relu')(input_1)
        hidden_2 = keras.layers.Dense(128, activation='relu')(hidden_1)
        hidden_3 = keras.layers.Dense(128, activation='relu')(hidden_2)
        output_1 =  keras.layers.Dense(5, activation='linear')(hidden_3)
        model = keras.Model(inputs=input_1, outputs=output_1)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary ()
        
        return model 

    def get_qs(self ,state):
        # predict a q-value based on single transition. 
        state  = np.reshape(state, (1,31))             
        return self.model.predict(state)
    
    def get_batch_qs(self, state): 
        # predict a q-value based on a batch of the transitions.
        return self.model.predict(state)

    def train(self, termination_state ,minibatch): 

        current_state = np.array([sample[0] for sample in minibatch])

        current_qs_list = self.get_batch_qs(current_state)  # predict q(state) for every sample 
        # get state after action from minibatch and query NN model for Q values 
        new_current_state  = np.array([sample[2] for sample in minibatch])
        
        future_qs_list = self.target_model.predict(new_current_state)
        
        x = []              # State list (input date) 
        y = []              # Action (lable)

        for index, (current_state, rand_action, new_current_state, reward, done) in enumerate (minibatch): 
            
            if done is False: 
                max_future_q  = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else: 
                new_q = reward      # if we are done new_q = reward because there is no future q 

            # Update q-value 
            current_qs = current_qs_list[index]
            current_qs [rand_action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), epochs=100 ,batch_size=self.batch_size, shuffle=False, verbose=1)
        
        #update target network counter by every episode
        if termination_state: 
            self.target_update_counter += 1
        
        if self.target_update_counter > self.Update_target_every: 
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models/')
        self.model.save('models/training.h5')
        self.model.save('models/FNN model')
