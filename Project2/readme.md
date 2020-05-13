Hi there! 

In my project you find the following files:
1) Episode.py: This contains the functions required to run 1 episode of the Lunar Lander environment. It takes 2 keras neural networks as inputs along with additional parameters
2) DQN.py: This file reads a set of hyperparameters as input, builds a deep q netowrk and then runs 1,500 training episodes and 100 testing episodes using the networks.
3) Experiments.py: This file creates a list of hyperparameter sets. Then, for each set of hyperparameters, runs an experiment (1,500 training episodes + 100 testing episodes) and then outputs a file with the results to the output folder.

The output folder has text files of all the experiment results. It also has Charts.rmd an R markdown file that is used to generate all of the charts in the report.

Note that this project was completed using Python in Spyder. Charts were completed using RStudio. All packages used are listed at the top of each code file.
