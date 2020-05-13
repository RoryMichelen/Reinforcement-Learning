# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:22:58 2020

@author: Rory
"""
import numpy as np
import environment as envr

nStates=7

def TrainTilConvergence(trainingSet,l,a,epsilon):
# input a training set and hyperparameters, outputs a trained, W vector, WT
# This function will train a set until convergence if epsilon is positive. Otherwise, it will train the set once.
    delta=100
    wt=np.ones(nStates-2)*0.5
    c=False # c is used to end the function is delta is not converging
    t=0 # used for decaying alpha
    # negative alpha indicates to decay alpha. The absolute value of the negative alpha value will become the numerator of an x/T decay
    if a<0:
        decayAlpha=True
        a=a*-1
    else:
        decayAlpha=False
    # Negative epsilon indicates that you show each sequence once (i.e experiment 4)
    if epsilon==-1:
        wt=TrainSet(trainingSet,wt,l,a,True)
    else:
        while delta>epsilon and not c:
            t+=1
            d=0
            newW=TrainSet(trainingSet,wt,l,a,False)
            if delta==sum(abs(newW-wt)):
                d+=1
            elif delta>10000:
                #skip if delta is not converging
                c=True
                wt=float('NaN')
            delta=sum(abs(newW-wt))
            wt=newW
            if decayAlpha:
                a=a/t
            if d>5:
                print('ts')
    return(wt)

def TrainSet(trainingSet,w,l,a,updateBetweenSequences):
# trains a training set once.
    newW=w.copy()
    deltaW=np.zeros(nStates-2)
    for seq in trainingSet:
        p=GetPredictions(seq,newW)
        deltaW+=TrainSequence(seq,l,a,p)
        if updateBetweenSequences:
            # for experiment 4, update deltaW between sequences
            newW+=deltaW
            deltaW=np.zeros(nStates-2)
    if not updateBetweenSequences:
        newW+=deltaW
    return(newW)
        
def TrainSequence(seq,l,a,p):
# trains the sequence with the help of the below functions
    deltaW=np.zeros(nStates-2)
    for t in range(len(seq)-1):
        deltaW+=GetDeltaWAtT(seq,t,a,l,p)
    return(deltaW)

def GetDeltaWAtT(seq,t,a,l,p):
    weight=GetWeightAtT(seq,t,l)
    deltaW=a*(p[t+1]-p[t])*weight
    return(deltaW)
    
def GetPredictions(seq,w):
    p=np.zeros(len(seq))
    for t in range(len(seq)-1):
        state=seq[t]
        p[t]=w[envr.StateToInt(state,nStates)-1]
    state=seq[len(seq)-1]
    p[len(seq)-1]=state
    return(p)

def GetWeightAtT(seq,t,l):
    weight=np.zeros(nStates-2)
    for k in range(t+1):
        weight+=(l**(t-k))*seq[k]
    return(weight)









        
        
    