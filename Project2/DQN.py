# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:18:23 2020

@author: Rory
"""
# Given a set of parameters, this file implements the DQN algorithm, trains a model and then tests it

from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from collections import deque
import gym
import copy
import Episode as ep
import matplotlib.pyplot as plt
import time
import numpy as np

def Run_Experiment(params):
    env = gym.make('LunarLander-v2')
    Q=Create_Network(env,params.num_hidden,params.learning_rate)
    Q_target=copy.deepcopy(Q)
    memory=deque([],maxlen=int(params.memory_size))
    outputs=([],[])
    episode_start=time.time()
    for i in range(1500):
        start=time.time()
        ep.Run_Episode(Q,Q_target,params,memory,outputs,True)
        params.Update_Epsilon()
        duration=np.round((time.time() - start),0)
        reward=outputs[0][i]
        steps=outputs[1][i]
        print(f" I:{i}, Time:{duration}, Rew:{reward},steps:{steps}")
        if i%params.target_update_freq==0:
            Q_target=copy.deepcopy(Q)
    for i in range(100):
        ep.Run_Episode(Q,Q_target,params,memory,outputs,False)

    episode_duration=np.round((time.time() - episode_start),0)
    rewards,num_steps=outputs
    return(rewards,num_steps,episode_duration)

def Create_Network(env,num_hidden,learning_rate):
    Q = Sequential()        
    Q.add(Dense(num_hidden, input_dim=env.observation_space.shape[0], activation="relu"))
    Q.add(Dense(num_hidden, activation="relu"))
    Q.add(Dense(env.action_space.n))
    Q.compile(loss="mean_squared_error",optimizer=Adam(lr=learning_rate))
    return(Q)



        

