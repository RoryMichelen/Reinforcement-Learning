# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:51:13 2020

@author: Rory
"""

import numpy as np

def GetTrainingSet(nSequences,nStates,startState):
    trainingSet=[]
    for i in range(0,nSequences):
        newSequence=GetSequence(nStates,startState)
        trainingSet.append(newSequence)
        
    return(trainingSet)

def GetSequence(nStates,startState):
    hasTerminated=False
    sequence=[startState]
    while not hasTerminated:
        sequence.append(TakeRandomStep(sequence,nStates))
        currentState=GetLastState(sequence)
        if isinstance(currentState,int):
            hasTerminated=True
#    if len(sequence)>10:
#        sequence=GetSequence(nStates,startState)
    
    return(sequence)

def TakeRandomStep(sequence,nStates):
    lastState=GetLastState(sequence)
    lastStateInt=StateToInt(lastState,nStates)
    if(np.random.rand(1)[0]>=0.5):
        newStateInt=lastStateInt+1
    else:
        newStateInt=lastStateInt-1
    return(IntToState(newStateInt,nStates))
       
def GetLastState(sequence):
    if(isinstance(sequence,list)):
        state=sequence[len(sequence)-1]
    else:
        state=sequence
    return(state)
    
def StateToInt(state,nStates):
# Convert a state (represented as a 1-hot encoded vector) to an integer
    if isinstance(state,int):
        if state==1:
            stateInt=nStates-1
        elif state==0:
            stateInt=0
        else:
            print('A non 0/1 integer was passed as a state')
    else:
        stateInt=np.where(state==1)[0][0]+1
    return(stateInt)

def IntToState(stateInt,nStates):
# Convert an integer to a 1-hot encoded vector.
    if stateInt==nStates-1:
        state=1
    elif stateInt==0:
        state=0
    else:
        state=np.zeros(nStates-2)
        state[stateInt-1]=1
    return(state)

