import numpy as np
import matplotlib.pyplot as plt
import pandas

# Function that creates the X matrix as defined for fitting our model
def create_X(x,deg):
    X = np.ones((len(x),deg+1))
    for i in range(1,deg+1):
        X[:,i] = x**i
    return X

# Function for predicting the response
def predict_y(x,beta):
    return np.dot(create_X(x,len(beta)-1),beta)

# Function for fitting the model
def fit_beta(df,deg):
    return np.linalg.lstsq(create_X(df.x,deg),df.y,rcond=None)[0]

# Function for computing the MSE
def mse(y,yPred):
    return np.mean((y-yPred)**2)

# Loading training, validation and test data
dfTrain = pandas.read_csv('Data_Train.csv')
dfVal = pandas.read_csv('Data_Val.csv')
dfTest = pandas.read_csv('Data_Test.csv')

############ TRAINING A MODEL

# Fitting model
error = 100;
degOpt = 0.0;
for deg in range(1,20):
    X = create_X(dfTrain.x,deg)
    beta = fit_beta(dfTrain,deg)
    yPredVal = predict_y(dfVal.x,beta)
    err = mse(dfVal.y,yPredVal)
    if err < error:
        error = err
        degOpt = deg

print("Optimal degree selected:", degOpt)
# Concatenating data training and validation data frames
df = dfTrain.append(dfVal)

# Fit model using the optimal degree found in the previous cell
X = create_X(df.x, degOpt)
beta = fit_beta(df, degOpt)

# Compute and print training error
yPredTrain = predict_y(df.x,beta)
err = mse(df.y,yPredTrain)
print('Training Error = {:2.3}'.format(err))

# Compute and print testing error
yPredTest = predict_y(dfTest.x,beta)
err = mse(dfTest.y,yPredTest)
print('Testing Error = {:2.3}'.format(err))

#Plotting the graph
x = np.linspace(0,1,100)
y = predict_y(x,beta)

plt.plot(x,y,'b-',df.x,df.y,'r.')
plt.show()