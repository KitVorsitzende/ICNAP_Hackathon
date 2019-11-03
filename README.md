# ICNAP_Hackathon
Task3/Process Monitoring/Classification
This the task 3 of ICNAP Hackathon in Aachen on 25-27 Okt 2019
Try in the competition
Main goal is to classify the given time series data, the labeled training data is provided.
During the 33 hours competition time, we managed to apply the Random Tree Classificator to classify the unkonwn data, 
to decide which types (0,1,2) it belongs, and how many gaussian distributions are mixtured in it's probablity distribution.

We tried to divide the whole time series data, and calculate the mean, variance and 
10,25,50,75,90 percentile of the divided groups. Those vaules of each group were used as the features to feed into the
Randomtree.

In the end, the both subtask we have achieved a accuracy of 80%.


New Try a week later
A week later, I reconsidered the problem, and think a neuroal network might also works well for the data.
Since the time series data are different in length, I calculate the min length of all the training and testing data, 
and have only taken the min length into account.

All the average values of time series data are eliminated and normalized of area [-1,1]. 
All the 900 data are given into a simplest NN as input. And the accuracy reaches over 95% for first subtask, 
the classification of types (0,1,2). The accuracy reaches over 82% for second subtask.

Can be done: Deeper NN with more hidden layers

I tried also to feed all the 900 datas as features to the Radom Tress, it seems works pretty well for the first subtask,
if the n_estimators and max_depth are adapted, a accuracy of nearly 100% can be reached. 
But it worked poorly for the second task with about only 10% accuracy.
