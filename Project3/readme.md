# 7642Spring2020rmichelen3

Hello! Welcome to my Correlated Q-Learning Project. Some details are below:

The following project was developed mostly using Spyder. The most important files are listed below:

1) main.py executes all four experiments and reads from the files below
2) env.py implements the Soccer Environment as a class as well as a class for each state in the environment
3) agent.py implements the agent class. 
4) Q.py implements a parent class "Q" which has four child classes, one for each of the four algorithms. These classes have all of the important mechanisms of Q-Learning (i.e. selecting an action, updating the Q table). Each agent has an instance of a Q class as an attribute

When Main.py is finished running, it outputs two files to the "output_files" folder. The "Charts.rmd" file reads these files and produces the 5 figures in the report. This is an R Markdown file.

The Parameter Files folder has two important csv files that are read by the above scripts. Most importantly, is "agent_params.csv". This is read by main.py and contains the values of the hyperparameters for all experiments ran. Less important is the "transitions.csv" file which is read by env.py. This file specifies the transitions between states for each state action pair. 

In summary, if you want to reproduce the results, just run "Main.py" to produce the output files, then run "Charts.rmd" to produce the graphs.
