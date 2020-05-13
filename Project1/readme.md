Hi there!
My project was developed using anaconda and spyder.

To generate the charts, run main.py, the run lines 41-50 in ipython console, each chunk separately.

The other 3 files included are:
1) Experiments.py: It contains two functions, one for each experiment. Inputs are the parameters of the experiment such as lambda values, alpha values, convergence criteria
2) Agent2.py: is used to run variations of TD lambda as implemented in Sutton's paper. Functions from this script are called from experiments.py to run TD lambda for various hyperparameters
3) Environment.py contains code related to training set generation (i.e. creating random walks). Functions from this file are called from main.py

Enjoy!
