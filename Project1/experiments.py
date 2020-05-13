# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:04:28 2020

@author: Rory
"""
import pandas as pd
import agent2 as agent

def RunExp1(lambdas,alphas,deltas,trainingSets,z):
    output=pd.DataFrame({"l":[],"a":[],"d":[],"rmse":[]})
    for l in lambdas:
        for a in alphas:
            rmses=[]
            for d in deltas: 
                for t in trainingSets:
                    w=agent.TrainTilConvergence(t,l,a,d)
                    rmse=(sum((z-w)**2)/len(z))**0.5
                    rmses.append(rmse)
                output=output.append(pd.DataFrame({"l":[l],"a":[a],"d":[d],"rmse":[sum(rmses)/len(rmses)]}))
    return(output)

def RunExp2(lambdas,alphas,trainingSets,z):
    output=pd.DataFrame({"l":[],"a":[],"rmse":[]})
    for l in lambdas:
        for a in alphas:
            rmses_exp2=[]
            for t in trainingSets:
                w=agent.TrainTilConvergence(t,l,a,-1)
                rmse=(sum((z-w)**2)/len(z))**0.5
                rmses_exp2.append(rmse)
            output=output.append(pd.DataFrame({"l":[str(l)],"a":[a],"rmse":[sum(rmses_exp2)/len(rmses_exp2)]}))
    return(output)
