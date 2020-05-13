# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:43:08 2020

@author: Rory
"""
import numpy as np

class environment:
    def __init__(self,state,is_test=False):
        np.random.seed(1)
        self.num_states=8
        self.num_actions=5
        self.state=state
        self.is_test=is_test
        self.transitions=np.genfromtxt('Parameter Files/transitions.csv', delimiter=',').astype(int)
        self.done=False
    
    def take_action(self,action_a,action_b):
        if self.done:
            self.state.reset()
            self.done=False
        next_state_a=int(self.transitions[self.state.a,action_a])
        next_state_b=int(self.transitions[self.state.b,action_b])
        # Interpet a collison as each player's actions result in occupying the same square
        # So any sequence where player 1 moves to player 2's followed by player 2 moving out is allowed
        if next_state_a==next_state_b:
            a_goes_first=np.random.rand()>=0.5 if self.is_test==False else True
            if a_goes_first:
                if self.state.b!=next_state_b:
                    self.state.a=next_state_a
                    if self.state.ball=='b':
                        self.state.ball='a'
            else:
                if self.state.a!=next_state_a:
                    self.state_b=next_state_b
                    if self.state.ball=='a':
                        self.state.ball='b' 
        else:
            self.state.a=next_state_a
            self.state.b=next_state_b
        
        
        return(self.get_action_outcome()) #next_state,reward_a,reard_b,done
    def take_action_no_swaps(self,action_a,action_b):
        if self.done:
            self.state.reset()
            self.done=False
        next_state_a=int(self.transitions[self.state.a,action_a])
        next_state_b=int(self.transitions[self.state.b,action_b])
        
        a_goes_first=np.random.rand()>=0.5 if self.is_test==False else True
        if a_goes_first: 
            # if A goes first and collides with B then A loses the ball 
            if next_state_a==self.state.b:
                if self.state.ball=='a':
                    self.state.ball='b' 
            else:
                self.state.a=next_state_a
            # if B goes second and collides with A then B loses the ball
            if next_state_b==self.state.a:
                if self.state.ball=='b':
                    self.state.ball='a' 
            else:
                self.state.b=next_state_b
        else:
            # if B goes first and collides with A then B loses the ball
            if next_state_b==self.state.a:
                if self.state.ball=='b':
                    self.state.ball='a' 
            else:
                self.state.b=next_state_b
            
            if next_state_a==self.state.b:
                if self.state.ball=='a':
                    self.state.ball='b' 
            else:
                self.state.a=next_state_a
                
        return(self.get_action_outcome()) #next_state,reward_a,reard_b,done
        
    def get_action_outcome(self):
        reward_a=0
        reward_b=0
        if self.state.ball=='a':
            if self.state.a==3 or self.state.a==7:
                reward_a=-100
                self.done=True
            elif self.state.a==0 or self.state.a==4:
                reward_a=100
                self.done=True
            reward_b=-reward_a
        elif self.state.ball=='b':
            if self.state.b==0 or self.state.b==4:
                reward_b=-100
                self.done=True
            elif self.state.b==3 or self.state.b==7:
                reward_b=100
                self.done=True
            reward_a=-reward_b
        
        return(self.state,reward_a,reward_b,self.done)

class state:
    def __init__(self,state_a=2,state_b=1,state_ball='b'):
        self.a=state_a
        self.b=state_b
        self.ball=state_ball
        self.a_init=state_a
        self.b_init=state_b
        self.ball_init=state_ball
    
    def reset(self):
        self.a=self.a_init
        self.b=self.b_init
        self.ball=self.ball_init
    
    def ball_int(self):
        return 1 if self.ball=='b' else 0
        





            
            
                
            

            