'''
Created on 21 Aug 2018

@author: seanv
'''
from Data import validate
from sklearn.metrics import mean_squared_error
from math import sqrt
def test_function():
    X = validate.getdataset().values
    X.astype('float32')

    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]

# walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
    # predict
        yhat = history[-1]
        predictions.append(yhat)
    # observation
        obs = test[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)