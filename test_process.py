
def test_run(X_test, Y_test, W_test_1, W_test_2, W_test_3, b_test_1, b_test_2, b_test_3):
  
  import numpy as np
  
  #Linear function in first layer

  Z_test_1 = np.dot(W_test_1,X_test) + b_test_1

  #Tanh activation function in first layer

  A_test_1 = np.tanh(Z_test_1)

  #Linear function in second layer

  Z_test_2 = np.dot(W_test_2, A_test_1) + b_test_2

  #Tanh activation function in second layer

  A_test_2 = np.tanh(Z_test_2)

  #Linear function in third layer

  Z_test_3  = np.dot(W_test_3, A_test_2) + b_test_3

  #SoftMax function in third layer

  t = np.exp(Z_test_3)
  A_test_3 = t/(np.sum(t, axis = 0))

  #Accuracy calculation based on output of SoftMax

  Accuracy = np.sum(Y_test*A_test_3/Y_test.shape[1])

  #Cost function updated with L2 regularization
  Cost = -(1/Y_test.shape[1])*np.sum(np.multiply(np.log(A_test_3),Y_test) + (1-Y_test)*np.log(1-A_test_3))

  #Report cost and accuracy

  print("The cost is %6.4f, and the Accuracy %4.2f %%" % (Cost, 100*Accuracy))
  
  return A_test_3