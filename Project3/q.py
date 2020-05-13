# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:43:54 2020

@author: Rory
"""
import numpy as np
import random
from cvxopt import matrix, solvers
from scipy.linalg import block_diag


class q:
    def __init__(self,env,train_control):
        self.num_states=env.num_states
        self.num_actions=env.num_actions
        self.train_control=train_control
        np.random.seed(1)
        random.seed(1)
    
    def choose_action(self,env):#,state):
        if np.random.rand()>self.train_control.epsilon:
            best_options=np.argwhere(self.q_table[env.state.a][env.state.b][env.state.ball_int()]==np.amax(self.q_table[env.state.a][env.state.b][env.state.ball_int()]))
            if np.size(best_options,1)==2:
                return(random.choice(best_options)[1])
            else:
                return(random.choice(best_options))
        else:
            return(np.random.randint(env.num_actions))
            
    def learn_single(self,state,action,next_state_val,reward,done):
        curr_val=self.q_table[state.a][state.b][state.ball_int()][action]
        self.q_table[state.a][state.b][state.ball_int()][action]=((1-self.train_control.alpha)*curr_val)+(self.train_control.alpha*((1-self.train_control.gamma)*reward+self.train_control.gamma*next_state_val))
    
    def learn_joint(self,state,action_primary,action_secondary,next_state_val,reward,done):
        curr_val=self.q_table[state.a][state.b][state.ball_int()][action_secondary][action_primary]
        self.q_table[state.a][state.b][state.ball_int()][action_secondary][action_primary]=((1-self.train_control.alpha)*curr_val)+(self.train_control.alpha*((1-self.train_control.gamma)*reward+self.train_control.gamma*next_state_val))

    def get_experiment_q_single(self):
        return self.q_table[2][1][1][1]
    
    def get_experiment_q_joint(self):
        return self.q_table[2][1][1][4][1]


class q_standard(q):
    def __init__(self,env,train_control):
        super().__init__(env,train_control)
        self.q_table=np.zeros((self.num_states,self.num_states,2,self.num_actions))
    
    def choose_action(self,env,policy):#,state):
        return super().choose_action(env)#,state)
         
    def learn(self,state,action_primary,action_secondary,next_state,reward,done):
        next_state_val=np.max(self.q_table[next_state.a][next_state.b][next_state.ball_int()]) #if not done else 0
        super().learn_single(state,action_primary,next_state_val,reward,done)
    
    def get_experiment_q(self):
        return super().get_experiment_q_single()
        
class q_friend(q):
    def __init__(self,env,train_control):
        super().__init__(env,train_control)
        self.q_table=np.zeros((self.num_states,self.num_states,2,self.num_actions,self.num_actions))
        
    def choose_action(self,env,policy):#,state):
        return super().choose_action(env)#,state)
   
    def learn(self,state,action_primary,action_secondary,next_state,reward,done):
        next_state_val=np.max(self.q_table[next_state.a][next_state.b][next_state.ball_int()])# if not done else 0
        super().learn_joint(state,action_primary,action_secondary,next_state_val,reward,done)
        
    def get_experiment_q(self):
        return super().get_experiment_q_joint()

class q_foe(q):
    def __init__(self,env,train_control):
        super().__init__(env,train_control)
        self.q_table=np.ones((self.num_states,self.num_states,2,self.num_actions,self.num_actions))

        self.c = matrix([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.a = matrix([[0.0],[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.b = matrix(1.0)
        self.V=np.ones((self.num_states,self.num_states,2))
        self.piN=[]
        self.piS=[]
        self.piE=[]
        self.piW=[]
        self.piSt=[]               
        
    def choose_action(self,env,policy):#state):
        if np.random.rand()>self.train_control.epsilon:
            return(np.random.choice([0,1,2,3,4],p=policy))
        else:
            return(np.random.randint(env.num_actions))
    
    def get_policy(self,env,opponent):
        solvers.options['show_progress'] = False
        self.g = matrix(np.append(np.append(np.ones((5,1)), -1*self.q_table[env.state.a][env.state.b][env.state.ball_int()], axis=1), np.append(np.zeros((5,1)), -1*np.eye(5), axis=1), axis=0))
        sol=solvers.lp(c=self.c, G=self.g, h=self.h, A=self.a, b=self.b)
        policy=np.abs(sol['x'][1:]).reshape(5)/sum(np.abs(sol['x'][1:]).reshape(5))
        val=sol['x'][0]  
        
        if env.state.a==2 and env.state.b==1 and env.state.ball=='b':
           self.piN.append(policy[0])
           self.piS.append(policy[1])
           self.piE.append(policy[2])
           self.piW.append(policy[3])
           self.piSt.append(policy[4])
        return(policy,val)
        
    def learn(self,state,action_primary,action_secondary,next_state,reward,done):
        next_state_val=self.V[next_state.a,next_state.b,next_state.ball_int()]# if not done else 0
        super().learn_joint(state,action_primary,action_secondary,next_state_val,reward,done)
        
    def get_experiment_q(self):
        return super().get_experiment_q_joint()
    
class q_uce(q):
    def __init__(self,env,train_control):
        super().__init__(env,train_control)
        self.q_table=np.ones((self.num_states,self.num_states,2,self.num_actions,self.num_actions))

        self.V=np.ones((self.num_states,self.num_states,2))
        
        self.piN=[]
        self.piS=[]
        self.piE=[]
        self.piW=[]
        self.piSt=[]   
    
    def choose_action(self,env,policy):
        if np.random.rand()>self.train_control.epsilon:
            return(np.random.choice([0,1,2,3,4],p=policy))
        else:
            return(np.random.randint(env.num_actions))
    
    def get_policy(self,env,opponent):
        solvers.options['show_progress'] = False
        
        q_self_state=self.q_table[env.state.a,env.state.b,env.state.ball_int()]
        q_opp_state=opponent.q.q_table[env.state.a,env.state.b,env.state.ball_int()]

        c=matrix(-(q_self_state+q_opp_state).reshape(25))
        a = matrix(np.ones((1, 25)))
        b = matrix(1.0)

        g_probs=-np.eye(25)
        h_probs=np.zeros((25))

        g_self_0=-np.vstack((q_self_state[:,0]-q_self_state[:,1],q_self_state[:,0]-q_self_state[:,2],q_self_state[:,0]-q_self_state[:,3],q_self_state[:,0]-q_self_state[:,4]))
        g_self_1=-np.vstack((q_self_state[:,1]-q_self_state[:,0],q_self_state[:,1]-q_self_state[:,2],q_self_state[:,1]-q_self_state[:,3],q_self_state[:,1]-q_self_state[:,4]))
        g_self_2=-np.vstack((q_self_state[:,2]-q_self_state[:,1],q_self_state[:,2]-q_self_state[:,0],q_self_state[:,2]-q_self_state[:,3],q_self_state[:,2]-q_self_state[:,4]))
        g_self_3=-np.vstack((q_self_state[:,3]-q_self_state[:,1],q_self_state[:,3]-q_self_state[:,2],q_self_state[:,3]-q_self_state[:,0],q_self_state[:,3]-q_self_state[:,4]))
        g_self_4=-np.vstack((q_self_state[:,4]-q_self_state[:,1],q_self_state[:,4]-q_self_state[:,2],q_self_state[:,4]-q_self_state[:,3],q_self_state[:,4]-q_self_state[:,0]))

        g_self=block_diag(g_self_0,g_self_1,g_self_2,g_self_3,g_self_4)
        h_self=np.zeros((20))

        g_opp_0=-np.vstack((q_opp_state[:,0]-q_opp_state[:,1],q_opp_state[:,0]-q_opp_state[:,2],q_opp_state[:,0]-q_opp_state[:,3],q_opp_state[:,0]-q_opp_state[:,4]))
        g_opp_1=-np.vstack((q_opp_state[:,1]-q_opp_state[:,0],q_opp_state[:,1]-q_opp_state[:,2],q_opp_state[:,1]-q_opp_state[:,3],q_opp_state[:,1]-q_opp_state[:,4]))
        g_opp_2=-np.vstack((q_opp_state[:,2]-q_opp_state[:,1],q_opp_state[:,2]-q_opp_state[:,0],q_opp_state[:,2]-q_opp_state[:,3],q_opp_state[:,2]-q_opp_state[:,4]))
        g_opp_3=-np.vstack((q_opp_state[:,3]-q_opp_state[:,1],q_opp_state[:,3]-q_opp_state[:,2],q_opp_state[:,3]-q_opp_state[:,0],q_opp_state[:,3]-q_opp_state[:,4]))
        g_opp_4=-np.vstack((q_opp_state[:,4]-q_opp_state[:,1],q_opp_state[:,4]-q_opp_state[:,2],q_opp_state[:,4]-q_opp_state[:,3],q_opp_state[:,4]-q_opp_state[:,0]))

        g_opp=block_diag(g_opp_0,g_opp_1,g_opp_2,g_opp_3,g_opp_4)
        h_opp=np.zeros((20))

        g=matrix(np.vstack((g_probs,g_self,g_opp)))
        h=matrix(np.hstack((h_probs,h_self,h_opp)))
        
        try:
            sol = solvers.lp(c=c, G=g, h=h, A=a, b=b)
            if sol['x'] is not None:
                policy = np.abs(np.array(sol['x']).reshape((5, 5))) / sum(np.abs(sol['x']))
                val_self = np.sum(policy*q_self_state)
                val_opp = np.sum(policy *q_opp_state.T)
                
                if env.state.a==2 and env.state.b==1 and env.state.ball=='b':
                    self.piN.append(policy[0]+policy[5]+policy[10]+policy[15]+policy[20])
                    self.piS.append(policy[1]+policy[6]+policy[11]+policy[16]+policy[21])
                    self.piE.append(policy[2]+policy[7]+policy[12]+policy[17]+policy[22])
                    self.piW.append(policy[3]+policy[8]+policy[13]+policy[18]+policy[23])
                    self.piSt.append(policy[4]+policy[9]+policy[14]+policy[19]+policy[24])  
            else:
                policy = None
                val_self = None
                val_opp = None
        except:
            policy = None
            val_self = None
            val_opp = None

        return policy, val_self, val_opp    

    def learn(self,state,action_primary,action_secondary,next_state,reward,done):
        next_state_val=self.V[next_state.a,next_state.b,next_state.ball_int()]# if not done else 0
        super().learn_joint(state,action_primary,action_secondary,next_state_val,reward,done)
        
    def get_experiment_q(self):
        return super().get_experiment_q_joint()