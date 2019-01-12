import numpy as np
import numpy.matlib as matlib
import scipy as sp
import sqlite3
import matplotlib.pyplot as plt
import parameter_reporting
import retrieve_parameters
import training_process
import test_process
import database_operations

conn = sqlite3.connect('characters_final.sqlite')
cur = conn.cursor()

#Remove previous test results
database_operations.old_test_data_removal(cur)

#Regularization hyperparameter (lambda)
lmbd = 0.03575

#Arrays for collecting plot data
iteration_series = []
cost_series = []
accuracy_series = []


#Initiate training parameters (weights Wi and biases bi)
W_train_1 = np.empty([15,28])
W_train_2 = np.empty([15,15])
W_train_3 = np.empty([10,15])
b_train_1 = np.zeros((15,1))
b_train_2 = np.zeros((15,1))
b_train_3 = np.zeros((10,1))
learning_rate = 0.0225
i=0
'''
Fetch preset parameters (W_train_1...W_train_3) from output files. The purpose of this is to avoid the instability caused by random parameter initialization.
'''
retrieve_parameters.parameter_fetch(W_train_1, "W1_recorded.txt")
retrieve_parameters.parameter_fetch(W_train_2, "W2_recorded.txt")
retrieve_parameters.parameter_fetch(W_train_3, "W3_recorded.txt")
print("Training parameters have been initialized...")

#Fetch training data from database
X_train, Y_train = database_operations.data_retrieval(cur)

#Run training process
W_train_1, W_train_2, W_train_3, b_train_1, b_train_2, b_train_3, iteration_series, cost_series, accuracy_series, A_train_3 = training_process.training(X_train, Y_train, W_train_1, W_train_2, W_train_3, b_train_1, b_train_2, b_train_3, lmbd, learning_rate, iteration_series, cost_series, accuracy_series)

#Write parameters for weight matrix W1:
parameter_reporting.writer(W_train_1, "W1_trained.txt")
parameter_reporting.writer(W_train_2, "W2_trained.txt")
parameter_reporting.writer(W_train_3, "W3_trained.txt")
parameter_reporting.writer(b_train_1, "b1_trained.txt")
parameter_reporting.writer(b_train_2, "b2_trained.txt")
parameter_reporting.writer(b_train_3, "b3_trained.txt")


#Write training results to database:
database_operations.training_data_write(cur,conn, A_train_3)


#Generate plot of training process
color='green'
fig, ax1 = plt.subplots()
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Accuracy (%)', color = color)
ax1.plot(iteration_series, accuracy_series, color = color)
ax1.tick_params(axis = 'y', labelcolor=color)
ax2=ax1.twinx()

color ='red'
ax2.set_ylabel('Cost', color=color)
ax2.plot(iteration_series, cost_series, color = color)
ax2.tick_params(axis = 'y', labelcolor = color)

plt.savefig('training_plot.png')

#Pause before testing
prompt=False
while prompt==False:
  prompt=input("Press any key to continue...")



#Initiate test parameters (weights Wi and biases bi)
W_test_1 = np.empty([15,28])
W_test_2 = np.empty([15,15])
W_test_3 = np.empty([10,15])
b_test_1 = np.empty([15,1])
b_test_2 = np.empty([15,1])
b_test_3 = np.empty([10,1])


#Fetch parameters from output files
retrieve_parameters.parameter_fetch(W_test_1, "W1_trained.txt")
retrieve_parameters.parameter_fetch(W_test_2, "W2_trained.txt")
retrieve_parameters.parameter_fetch(W_test_3, "W3_trained.txt")
retrieve_parameters.parameter_fetch(b_test_1, "b1_trained.txt")
retrieve_parameters.parameter_fetch(b_test_2, "b2_trained.txt")
retrieve_parameters.parameter_fetch(b_test_3, "b3_trained.txt")
print("Trained parameters have been retrieved and parameters are ready for testing...\n\n\n")

#Fetch test data from database
X_test, Y_test = database_operations.test_data_retrieval(cur)

#Run test using saved parameters
A_test_3 = test_process.test_run(X_test, Y_test, W_test_1, W_test_2, W_test_3, b_test_1, b_test_2, b_test_3)

#Save test results in database
database_operations.test_data_write(A_test_3, cur, conn)

#Print out predictions
database_operations.report_predictions(cur)









