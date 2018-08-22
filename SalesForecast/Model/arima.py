'''
Created on 21 Aug 2018

@author: seanv
'''
from Data import validate
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from matplotlib import pyplot
import numpy
from main import variables

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def saveArima(p, d, q, bias):
    series = Series.from_csv(variables.datasetloc)
    X = series.values
    X = X.astype('float32')
    
    interval = 12
    diff = difference(X, interval)
    
    model = ARIMA(diff, order=(p,d,q))
    model_fit = model.fit(trend='nc', disp=0)
    # bias constant, could be calculated from in-sample mean residual
    bias = 165.904728
    # save model
    
    model_fit.save(variables.model)
    numpy.save(variables.bias, [bias])

def arimaModel(p, d, q, bias):
    
    #ADDED
    series = Series.from_csv(variables.datasetloc)
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
    # difference data
        interval = 12
        diff = difference(history, interval)
        # predict
        model = ARIMA(diff, order=(p,d,q))
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = bias + inverse_difference(history, yhat, interval)
        predictions.append(yhat)
    # observation
        obs = test[i]
        history.append(obs)
        #print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)
    saveArima(p, d, q, bias)
    # errors
    residuals = [test[i]-predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    #print(residuals.describe())
    