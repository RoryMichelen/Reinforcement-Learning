# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:24:32 2020

@author: Rory
"""
# this file runs a single episode of the lunar lander
from keras.models import Sequential 
from keras.layers import Dense, Activation
from random import sample
import numpy as np
import pandas as pd
import gym
import time

np.random.seed(1)

def Run_Episode(Q,Q_target,params,memory,outputs,is_training):
    env = gym.make('LunarLander-v2')
    state=Reshape_State(env.reset())
    num_steps=0
    total_reward=0
    done=False
    epsilon=params.epsilon if is_training else -1
    while not done:
        num_steps+=1
        action=Choose_Action(Q,state,epsilon)
        state_prime, reward, done, info=Take_Action(env,action)
        total_reward+=reward
        done=True if done==True or num_steps==params.max_steps else False
        if is_training:
            memory.appendleft((state,action,reward,state_prime,done))
            Update_Q(Q,Q_target,params.gamma,memory,params.batch_size)
        state=state_prime

    UpdateOutputs(outputs,total_reward,num_steps)
    
            
def Update_Q(Q,Q_target,gamma,memory,batch_size):
    memory_list=list(memory)
    if(len(memory_list)>=batch_size):
        batch=sample(memory_list,batch_size)
        # Expand batch into its components
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        state_primes = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])
        states = np.squeeze(states)
        state_primes = np.squeeze(state_primes)
        
        # Create a vector of the estimates for each state-action observation in batch
        target_action_values = rewards + gamma*(np.amax(Q_target.predict_on_batch(state_primes), axis=1))*(1-dones)
       
        #Create another vector that has all predictions for all actions from each state in the batch
        targets_full = Q.predict_on_batch(states)
        ind = np.array([i for i in range(batch_size)])
        
        # For each observed state, insert the estimate from Q-target. So that when training the unobserved actions are not updated.
        targets_full[[ind], [actions]] = target_action_values

        Q.fit(states, targets_full, epochs=1, verbose=0)
        
#Q.fit(np.array(states), np.array(learning_targets),verbose=0)

def Choose_Action(Q,state,epsilon):
    # Get Q values for all actions astatet state
    q_values=Q.predict(state)
    if np.random.random(1) < epsilon:
        action = np.random.randint(0,q_values.shape[1])
    else:
        action=np.argmax(q_values[0])
    return action

def UpdateOutputs(outputs,total_reward,num_step):
    total_rewards,num_steps=outputs
    total_rewards.append(total_reward)
    num_steps.append(num_step)
    

def Reshape_State(state):
    return(state.reshape(1,8))

def Take_Action(env,action):
    state_prime, reward, done, info=env.step(action)
    state_prime=Reshape_State(state_prime)
    return (state_prime, reward, done, info)

 
