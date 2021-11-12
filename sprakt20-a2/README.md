# sprakt20-a2


### Task 4

#### Training:

##### Batch gradient descent:
convergence margin = 0.001
learning rate = 1
iters = 48

##### Stochastic gradient descent:
n_epochs = 1e6
learning rate = 0.8

##### Mini batch gradient descent:
convergence margin = 0.003
learning rate = 0.001
iters = 2.5e4


#### Results:

num_words = 101917

accuracy = (predicted(Name)|Name + predicted(No Name)|No name) / num_words

precision(Name) = predicted(Name)|Name / predicted(Name)
precision(No Name) = predicted(No Name)|No Name / predicted(No Name)

recall(Name) = predicted(Name)|Name / Name
precision(No Name) = predicted(No Name)|No Name / No Name


##### Batch gradient descent - 2 features: 
accuracy = 13273 + 80674 / 99998 = 0.939

precision(Name) = 13273 / 17309 = 0.767
precision(No Name) = 80674 / 82689 = 0.976

recall(Name) = 13273 / 15288 = 0.868
precision(No Name) = 80674 / 84710 = 0.952

##### Stochastic gradient descent- 2 features:
accuracy = 12046 + 83831 / 99998 = 0.959

precision(Name) = 12046 / 12925 = 0.932
precision(No Name) = 83831 / 87073 = 0.963

recall(Name) = 12046 / 15288 = 0.788
precision(No Name) = 83831 / 84710 = 0.989

--> Reason for good results: run on 1e8 epochs. 

##### Minibatch gradient descent - 2 features: 
accuracy = 13273 + 80674 / 99998 = 0.939

precision(Name) = 13273 / 17309 = 0.767
precision(No Name) = 80674 / 82689 = 0.976

recall(Name) = 13273 / 15288 = 0.868
precision(No Name) = 80674 / 84710 = 0.952

--> Tends to bias No Name.

##### Minibatch gradient descent - 5 features: 
accuracy = 1352 + 84436 / 99998 = 0.858

precision(Name) = 1352 / 1626 = 0.8314
precision(No Name) = 84436 / 98372 = 0.858

recall(Name) = 1352 / 15288 = 0.088
precision(No Name) = 84436 / 84710 = 0.9997

--> Bad features are added. And probably the specific combination does not present the features well. 

##### Minibatch gradient descent - 3 features (added after_in): 
accuracy = 11205 + 83834 / 99998 = 0.950

precision(Name) = 11205 / 12072 = 0.928
precision(No Name) = 83843 / 87926 = 0.954

recall(Name) = 11205 / 15288 = 0.733
precision(No Name) = 83843 / 84710 = 0.9897

--> Fewer but better features.

