# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:14:48 2020

@author: Rory
"""

import experiments as expt
import numpy as np
import plotnine as plt
import environment as envr

np.random.seed(0)
z=np.array([1/6,2/6,3/6,4/6,5/6])
trainingSets=[]

for i in range(100):
    trainingSets.append(envr.GetTrainingSet(10,7,envr.IntToState(3,7)))
    
# Experiment 1
lambdas=[0,0.1,0.3,0.5,0.7,0.9,1]
alphas=np.arange(0.024,0.032,0.001)
output_exp1=expt.RunExp1(lambdas,alphas,[0.1],trainingSets,z)

bestAlpha=output_exp1.groupby('l').agg(rmse=('rmse',min)).reset_index('l')

#Experiment 2
lambdas=[0,0.3,0.8,1]
alphas=np.arange(0.0,1.0,0.05)
output_exp2=expt.RunExp2(lambdas,alphas,trainingSets,z)

# rerun experiment 2 with different lambda values. Python is new to me 
# so this was easier than filtering -__-
lambdas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

output_exp21=expt.RunExp2(lambdas,alphas,trainingSets,z)
bestAlpha2=output_exp21.groupby('l').agg(rmse=('rmse',min)).reset_index('l')

#Plots
# Figure 3
plt.ggplot(bestAlpha,plt.aes(x='l',y='rmse'))+plt.geom_line()+plt.geom_point()

# Figure 4
plt.ggplot(output_exp2,plt.aes(x='a',y='rmse',color=('l'))) \
    +plt.geom_line(plt.aes(group='l')) \
    +plt.geom_point()\
    +plt.coord_cartesian(xlim=(0,0.62),ylim=(0,0.7))

# Figure 5
plt.ggplot(bestAlpha2,plt.aes(x='l',y='rmse'))+plt.geom_line(plt.aes(group=1))+plt.geom_point()








