import numpy as np


def training(X_train, Y_train, W_train_1, W_train_2, W_train_3, b_train_1, b_train_2, b_train_3, lmbd, learning_rate, iteration_series, cost_series, accuracy_series):
  
  for iterations in range(16000):
    #Linear function in first layer
  
    Z_train_1 = np.dot(W_train_1,X_train) + b_train_1
    
    #Tanh activation function in first layer

    A_train_1 = np.tanh(Z_train_1)

    #Linear function in second layer

    Z_train_2 = np.dot(W_train_2, A_train_1) + b_train_2
   
    #Tanh activation function in second layer

    A_train_2 = np.tanh(Z_train_2)

    #Linear function in third layer

    Z_train_3  = np.dot(W_train_3, A_train_2) + b_train_3

    #SoftMax function in third layer

    t = np.exp(Z_train_3)
    A_train_3 = t/(np.sum(t, axis = 0))

    #Calculation of L2 regularization parameter

    W_train_1_sqr_F = np.sum(np.square(W_train_1))
    W_train_2_sqr_F = np.sum(np.square(W_train_2))
    W_train_3_sqr_F = np.sum(np.square(W_train_3))
    W_F = W_train_1_sqr_F + W_train_2_sqr_F + W_train_3_sqr_F

    L2_regularization = W_F*(lmbd/(2*Y_train.shape[1]))

    
    #Accuracy calculation based on output of SoftMax
    
    Accuracy = np.sum(Y_train*A_train_3/Y_train.shape[1])
    
    #Cost function

    Cost = -(1/Y_train.shape[1])*np.sum(np.multiply(np.log(A_train_3),Y_train) + (1-Y_train)*np.log(1-A_train_3))
    Cost = Cost + L2_regularization
    
    
    #Report number of iterations and cost current cost

    if iterations%1000 == 0:
        print("After", iterations, "iterations, the cost is %6.4f, and the accuracy %4.2f %%" % (Cost, 100*Accuracy))
        iteration_series.append(iterations)
        cost_series.append(Cost)
        accuracy_series.append(100*Accuracy)

    #Backprop for SoftMax

    dCost = A_train_3-Y_train
    
    #Calculation of gradients for level-3 parameters
    
    dW3 = np.dot(dCost, A_train_2.T)/Y_train.shape[1]
    db3 = np.sum(dCost, axis = 1, keepdims = True)/Y_train.shape[1]


    #Update of level-3 parameters using gradients

    W_train_3 = W_train_3*(1+(lmbd/Y_train.shape[1])) - learning_rate * dW3
    b_train_3 = b_train_3 - learning_rate * db3

    #Backprop for level-2 activation and linear functions
    
    d_g_2 =1-np.power(A_train_2, 2)
    dZ2 = np.dot(W_train_3.T, dCost)*d_g_2
    
    #Calculation of gradients for level-2 parameters
    
    dW2 = (1/Y_train.shape[1])*np.dot(dZ2,A_train_1.T)
    db2 = (1/Y_train.shape[1])*np.sum(dZ2, axis = 1, keepdims = True)

    #Update of level-2 parameters using gradients
    
    W_train_2 = W_train_2*(1+(lmbd/Y_train.shape[1])) - learning_rate * dW2
    b_train_2 = b_train_2 - learning_rate * db2

    #Backprop for level-1 activation and linear functions
    
    d_g_1 = 1 - np.power(A_train_1, 2)
    dZ1 = np.dot(W_train_2.T, dZ2)*d_g_1
    
    #Calculation of gradients for level-2 parameters
    
    dW1 = (1/Y_train.shape[1])*np.dot(dZ1,X_train.T)
    db1 = (1/Y_train.shape[1])*np.sum(dZ1, axis = 1, keepdims = True)

    #Update of level-2 parameters using gradients
    
    W_train_1 = W_train_1*(1+(lmbd/Y_train.shape[1])) - learning_rate * dW1
    b_train_1 = b_train_1 - learning_rate * db1
  
  print("Training complete after", iterations, "iterations\n")  
  return(W_train_1, W_train_2, W_train_3, b_train_1, b_train_2, b_train_3, iteration_series, cost_series, accuracy_series, A_train_3)
