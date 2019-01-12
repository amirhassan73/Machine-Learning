#Ensure no old test results are left 
def old_test_data_removal(cur):
  cur.execute("DROP TABLE IF EXISTS Training_Results")
  cur.execute("DROP TABLE IF EXISTS Test_Results")


#Fetch training data from database
def data_retrieval(cur):
  import numpy as np
  i = 0
  #Initiate training matrices X and Y
  Y_train_temp = np.empty([1000,1])
  X_train = np.empty([1000,28])
  
  cur.execute("CREATE TABLE IF NOT EXISTS Training_Results (ID INTEGER PRIMARY KEY AUTOINCREMENT, A FLOAT, C FLOAT, D FLOAT, E FLOAT, F FLOAT, G FLOAT, H FLOAT, I FLOAT, P FLOAT, R FLOAT)")

  Training = cur.execute("SELECT CLASS, X1 , X2 , X3 , X4 , X5 , X6 , X7 , X8 , X9 , X10 , X11 , X12 , X13 , X14 , X15 , X16 , X17 , X18 , X19 , X20 , X21 , X22 , X23 , X24 , X25 , X26 , X27 , X28 FROM Training_Data")

  #Populate training X matrix
  for data in Training:
    Y_train_temp[i]=data[0]
    X_train[i,1:28]=data[1:28]
    i+=1
  X_train = X_train.T

  #Populate training Y matrix
  Y_train = np.zeros((10,1000))
  for y in range(1000):
    Y_train[int(Y_train_temp[y]-1),y]=1
  print("Training data retrieved and training matrices ready for training to begin...\n\n\n")
  return(X_train, Y_train)

#Write training results to database:
def training_data_write(cur,conn, A_train_3):
  import numpy as np
  for x in range(A_train_3.shape[1]):
    predictions=np.round(A_train_3[:,x]*100,2)

    cur.execute("INSERT INTO Training_Results (A, C, D, E, F, G, H, I, P, R) VALUES (?,?,?,?,?,?,?,?,?,?)",(predictions[0],predictions[1],predictions[2],predictions[3],predictions[4],predictions[5],predictions[6],predictions[7],predictions[8],predictions[9]))
  conn.commit()

#Fetch test data from database
def test_data_retrieval(cur):
  import numpy as np
  i=0
  #Initiate test matrices X and Y
  Y_test_temp = np.empty([5000,1])
  X_test = np.empty([5000,28])
  cur.execute("CREATE TABLE IF NOT EXISTS Test_Results (ID INTEGER PRIMARY KEY AUTOINCREMENT, A FLOAT, C FLOAT, D FLOAT, E FLOAT, F FLOAT, G FLOAT, H FLOAT, I FLOAT, P FLOAT, R FLOAT)")

  Testing = cur.execute("SELECT CLASS, X1 , X2 , X3 , X4 , X5 , X6 , X7 , X8 , X9 , X10 , X11 , X12 , X13 , X14 , X15 , X16 , X17 , X18 , X19 , X20 , X21 , X22 , X23 , X24 , X25 , X26 , X27 , X28 FROM Test_Data")


  for data in Testing:
    Y_test_temp[i]=data[0]
    X_test[i,1:28]=data[1:28]
    i+=1
  X_test = X_test.T

  Y_test = np.zeros((10,5000))
  for y in range(5000):
    Y_test[int(Y_test_temp[y]-1),y]=1
  return(X_test, Y_test)

#Save test results in database
def test_data_write(A_test_3, cur, conn):
  import numpy as np
  for x in range(A_test_3.shape[1]):
    predictions=np.round(A_test_3[:,x]*100,2)

    cur.execute("INSERT INTO Test_Results (A, C, D, E, F, G, H, I, P, R) VALUES (?,?,?,?,?,?,?,?,?,?)",(predictions[0],predictions[1],predictions[2],predictions[3],predictions[4],predictions[5],predictions[6],predictions[7],predictions[8],predictions[9]))
  conn.commit()


#Report predictions

def report_predictions(cur):
  Predictions_A = "SELECT COUNT() FROM Test_Results WHERE A > 50"
  cur.execute(Predictions_A)
  print("Predictions for A:",cur.fetchone()[0])
  Predictions_C = "SELECT COUNT() FROM Test_Results WHERE C > 50"
  cur.execute(Predictions_C)
  print("Predictions for C:",cur.fetchone()[0])
  Predictions_D = "SELECT COUNT() FROM Test_Results WHERE D > 50.00"
  cur.execute(Predictions_D)
  print("Predictions for D:",cur.fetchone()[0])
  Predictions_E = "SELECT COUNT() FROM Test_Results WHERE E > 50.00"
  cur.execute(Predictions_E)
  print("Predictions for E:",cur.fetchone()[0])
  Predictions_F = "SELECT COUNT() FROM Test_Results WHERE F > 50.00"
  cur.execute(Predictions_F)
  print("Predictions for F:",cur.fetchone()[0])
  Predictions_G = "SELECT COUNT() FROM Test_Results WHERE G > 50.00"
  cur.execute(Predictions_G)
  print("Predictions for G:",cur.fetchone()[0])
  Predictions_H = "SELECT COUNT() FROM Test_Results WHERE H > 50.00"
  cur.execute(Predictions_H)
  print("Predictions for H:",cur.fetchone()[0])
  Predictions_I = "SELECT COUNT() FROM Test_Results WHERE I > 50.00"
  cur.execute(Predictions_I)
  print("Predictions for I:",cur.fetchone()[0])
  Predictions_P = "SELECT COUNT() FROM Test_Results WHERE P > 50.00"
  cur.execute(Predictions_P)
  print("Predictions for P:",cur.fetchone()[0])
  Predictions_R = "SELECT COUNT() FROM Test_Results WHERE R > 50.00"
  cur.execute(Predictions_R)
  print("Predictions for R:",cur.fetchone()[0])
