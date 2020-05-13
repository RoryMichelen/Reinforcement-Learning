# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:29:58 2020

@author: Rory
"""

import env
import numpy as np

def run_test_case(test):
    pass_test=True
    info=''
    t_state_a=test[0]
    t_state_b=test[1]
    t_state_ball='a' if test[2]==0 else 'b'
    t_action_a=test[3]
    t_action_b=test[4]
    t_next_state_a=test[5]
    t_next_state_b=test[6]
    t_next_state_ball = 'a' if test[7]==0 else 'b'
    t_reward_a=test[8]
    t_reward_b=test[9]
    t_done=True if test[10]==0 else False
    game=env.environment(t_state_a,t_state_b,t_state_ball,True)
    next_state_a,next_state_b,next_state_ball,reward_a,reward_b,done=game.take_action(t_action_a,t_action_b)
    if next_state_a!=t_next_state_a:
        pass_test=False
        info=info+f'next_state_a: test:{t_next_state_a},actual:{next_state_a}'
    if next_state_b!=t_next_state_b:
        pass_test=False
        info=info+f'next_state_b: test:{t_next_state_b},actual:{next_state_b}'
    if next_state_ball!=t_next_state_ball:
        pass_test=False
        info=info+f'next_state_ball: test:{t_next_state_ball},actual:{next_state_ball}'
    if reward_a!=t_reward_a:
        pass_test=False
        info=info+f'reward_a: test:{t_reward_a},actual:{reward_a}'
    if reward_b!=t_reward_b:
        pass_test=False
        info=info+f'reward_b: test:{t_reward_b},actual:{reward_b}'
    if done!=t_done:
        pass_test=False
        info=info+f'done: test:{t_done},actual:{done}'
    return pass_test,info

    
test_cases=np.genfromtxt('env_test.csv', delimiter=',').astype(int)
output=[]
for test in test_cases:
    output.append(run_test_case(test))
        
    

        