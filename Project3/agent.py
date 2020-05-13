# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:30:03 2020

@author: Rory
"""
import numpy as np
import q

class agent:
    def __init__(self,env,train_control):
        self.train_control=train_control
        self.opponent=None
        if self.train_control.q_type=='q':
            self.q=q.q_standard(env,train_control)
        elif self.train_control.q_type=='friend_q':
            self.q=q.q_friend(env,train_control)
        elif self.train_control.q_type=='foe_q':
            self.q=q.q_foe(env,train_control)
        elif self.train_control.q_type=='uce_q':
            self.q=q.q_uce(env,train_control)
        else:
            print('Invalid Agent Type!')

class train_control:
    def __init__(self,alpha,alpha_schedule,alpha_min,epsilon,epsilon_schedule,epsilon_min,gamma,q_type):
        self.alpha=alpha
        #self.alpha_schedule=alpha_schedule
        self.alpha_min=alpha_min
        self.alpha_schedule=(self.alpha_min/self.alpha)**(1/1000000)
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_schedule=(self.epsilon_min/self.epsilon)**(1/1000000)
        self.gamma=gamma
        self.q_type=q_type
    
    def decay_epsilon(self):
        self.epsilon*=self.epsilon_schedule
        self.epsilon=max(self.epsilon,self.epsilon_min)
    
    def decay_alpha(self):
        self.alpha*=self.alpha_schedule
        self.alpha=max(self.alpha,self.alpha_min)
        