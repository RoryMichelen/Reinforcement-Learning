# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:51:49 2020

@author: Rory
"""
# This file executes an "Experiment" for each combination of hyperparameters specified.
# An experiment includes a 1,500 episode training sequence followed by a 100 episode testing sequence

import numpy as np
import dqn 
import pandas as pd
import copy

class Params:
    def __init__(self,num_hidden=64,learning_rate=0.0005,target_update_freq=10,max_steps=1000,epsilon=1,epsilon_decay=0.995,gamma=0.99,memory_size=100000,batch_size=64,is_decay_mult=True,min_epsilon=0.01,experiment_name='Default',target_value="NA"):
        self.num_hidden=num_hidden
        self.learning_rate=learning_rate
        self.target_update_freq=target_update_freq
        self.max_steps=max_steps
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.gamma=gamma
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.is_decay_mult=is_decay_mult
        self.min_epsilon=min_epsilon
        self.experiment_name=experiment_name
        self.target_value=target_value
    
    def Run_Experiment(self):
        rewards,steps,duration=dqn.Run_Experiment(self)
        ind=np.arange(0,len(rewards))
        df = pd.DataFrame({"Experiment":self.experiment_name,"Val":self.target_value,"Trial":ind,"Reward" : rewards, "Steps" : steps, "RunTime" : duration})
        df.to_csv("Outputs/" +self.experiment_name+str(self.target_value)+".csv", index=False)
    
    def Update_Epsilon(self):
        if self.is_decay_mult:
            self.epsilon*=self.epsilon_decay
        else:
            self.epsilon-=self.epsilon_decay
        self.epsilon=min(self.epsilon,self.min_epsilon)
        
def Run_Experiment(param):
    param.Run_Experiment()
    
p=Params(experiment_name='Default_v2',batch_size=256)
p.Run_Experiment()


gammas=np.array((0.8,0.9,1))
epsilon_decays=np.array((0.75,0.85,1))
learning_rates=np.array((0.0001,0.001,0.002,0.01))
batch_sizes=np.array((256,512,1024))
memory_sizes=np.array((5000,20000,1000000))

max_steps=np.array((250,500,750))
num_hiddens=np.array((4,16,256))
target_update_freqs=np.array((1,50,100))

experiments=[]
experiments.append(Params())

p=Params(experiment_name='Gamma')
for val in gammas:
    p.gamma=val
    p.target_value=val
    experiments.append(copy.copy(p))

p=Params(experiment_name='epsilon_decay')
for val in epsilon_decays:
    p.epsilon_decay=val
    p.target_value=val
    experiments.append(copy.copy(p))

p=Params(experiment_name='learning_rates')
for val in learning_rates:
    p.learning_rates=val
    p.target_value=val
    experiments.append(copy.copy(p))

p=Params(experiment_name='batch_sizes')
for val in batch_sizes:
    p.batch_sizes=val
    p.target_value=val
    experiments.append(copy.copy(p))

experiments=np.array(experiments)
Run_All_Experiments=np.vectorize(Run_Experiment)
Run_All_Experiments(experiments)




    
    





    
    
    

    
