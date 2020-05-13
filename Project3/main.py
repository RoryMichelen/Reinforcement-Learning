# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:01:19 2020

@author: Rory
"""

import agent
import env
import pandas
import numpy as np
from scipy.linalg import block_diag

simulation_list=pandas.read_csv('Parameter Files/agent_params.csv')

def run_simulation(agent_a,agent_b,name,q_type):
    np.random.seed(1)
    errs=[]
    q=[]
    alpha=[]
    epsilon=[]
    tot_steps=0
    policy_a=np.ones(5)/5
    policy_b=np.ones(5)/5
    print(f'Starting {name}')
    while tot_steps<1000000:
        done=False
        num_steps=0
        while not done:
            num_steps+=1
            curr_state=env.state(envr.state.a,envr.state.b,envr.state.ball)
            
            action_a=agent_a.q.choose_action(envr,policy_a)#,envr.state.a)
            action_b=agent_b.q.choose_action(envr,policy_b)#,envr.state.b)
            next_state,reward_a,reward_b,done=envr.take_action_no_swaps(action_a,action_b)
            if num_steps>=1000:
                done=True
            prev_val_s=agent_a.q.get_experiment_q()
            
            agent_a.q.learn(curr_state,action_a,action_b,next_state,reward_a,done)
            agent_b.q.learn(curr_state,action_b,action_a,next_state,reward_b,done)
            
            if q_type == 'foe_q':       
                policy_a,val_a=agent_a.q.get_policy(envr,agent_b)
                policy_b,val_b=agent_b.q.get_policy(envr,agent_a)
                agent_a.q.V[envr.state.a,envr.state.b,envr.state.ball_int()]=val_a
                agent_b.q.V[envr.state.a,envr.state.b,envr.state.ball_int()]=val_b
            elif q_type=='uce_q':
                joint_policy,val_a,val_b=agent_a.q.get_policy(envr,agent_b)
                if val_a is not None:
                    agent_a.q.V[envr.state.a,envr.state.b,envr.state.ball_int()]=val_a
                    agent_b.q.V[envr.state.a,envr.state.b,envr.state.ball_int()]=val_b

            curr_val_s=agent_a.q.get_experiment_q()
            errs.append(abs(curr_val_s-prev_val_s))
            q.append(curr_val_s)
             
            alpha.append(agent_a.train_control.alpha)
            epsilon.append(agent_a.train_control.epsilon)
            
            agent_a.train_control.decay_alpha()
            agent_a.train_control.decay_epsilon()
            agent_b.train_control.decay_alpha()
            agent_b.train_control.decay_epsilon()
    
        tot_steps+=num_steps
        
        print(tot_steps)

    df = pandas.DataFrame(data={"name":name,"q":q,"errs": errs,"alpha":alpha,"epsilon":epsilon})
    return(df)


outputs=[]
pis=[]
for sim in simulation_list.iterrows():
    if sim[1][8]==0:
        envr=env.environment(env.state())
        agent_a=agent.agent(envr,agent.train_control(sim[1][0],sim[1][1],sim[1][2],sim[1][3],sim[1][4],sim[1][5],sim[1][6],sim[1][7]))
        agent_b=agent.agent(envr,agent.train_control(sim[1][0],sim[1][1],sim[1][2],sim[1][3],sim[1][4],sim[1][5],sim[1][6],sim[1][7]))
    
        agent_a.opponent=agent_b
        agent_b.opponent=agent_a
        
        outputs.append(run_simulation(agent_a,agent_b,sim[1][9],sim[1][7]))
        if(sim[1][7]=='foe_q' or sim[1][7]=='uce_q'):
            pis.append(pandas.DataFrame(data={"name":sim[1][7],"pi_N":agent_a.q.piN,"pi_S":agent_a.q.piS,"pi_E":agent_a.q.piE,"pi_W":agent_a.q.piE,"pi_St":agent_a.q.piSt}))
        
pandas.concat(outputs).to_csv("./output_files/output.csv", sep=',',index=True)
pandas.concat(pis).to_csv("./output_files/output_pi.csv", sep=',',index=True)