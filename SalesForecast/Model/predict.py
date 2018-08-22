'''
Created on 21 Aug 2018

@author: seanv
'''
from Data import validate
from pandas import Series
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
import matplotlib.lines as mlines
from math import sqrt
import numpy
from matplotlib import pyplot
from main import variables

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def predictRun(p, d, q):
    
    X = validate.getdataset().values.astype('float32')
    history = [x for x in X]
    months_in_year = 12
    y = validate.getvalidation().values.astype('float32')
    # load model
    model_fit = ARIMAResults.load(variables.model)
    bias = numpy.load(variables.bias)
    # make first prediction
    predictions = list()
    yhat = float(model_fit.forecast()[0])
    yhat = bias + inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    history.append(y[0])
    print('>Predicted=%.3f, Expected=%3.f, accuracy=%3.f' % (yhat, y[0], yhat/y[0]*100))
    # rolling forecasts
    for i in range(1, len(y)):
    # difference data
        months_in_year = 12
        diff = difference(history, months_in_year)
        # predict
        model = ARIMA(diff, order=(p,d,q))
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = bias + inverse_difference(history, yhat, months_in_year)
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%3.f, accuracy=%3.f' % (yhat, obs, yhat/obs*100))
        # report performance
    mse = mean_squared_error(y, predictions)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    
    red_line = mlines.Line2D([], [], color='red', label='Voorspelling')
    blue_line = mlines.Line2D([], [], color='blue', label='Werkelijkheid')
    pyplot.legend(handles=[red_line, blue_line])

    pyplot.title('Voorspelling verkoop wijn 1972')
    pyplot.ylabel('Hoeveelheid')
    pyplot.xlabel('Maand')
    pyplot.show()

def predictFuture():
    series = validate.getdatafile()
    months_in_year = 12
    model_fit = ARIMAResults.load(variables.model)
    bias = numpy.load(variables.bias)
    yhat = float(model_fit.forecast()[0])
    yhat = bias + inverse_difference(series.values, yhat, months_in_year)
    print('Predicted: %.3f' % yhat)
    pyplot.plot(series)
    pyplot.title('Lijn diagram tot aan voorspelling')
    pyplot.show()
    
    prediction = pd.DataFrame(yhat)
    prediction.to_csv(variables.predictionSave, mode='a', sep=',', header=False)
    prediction.to_csv(variables.datafileloc, mode='a', header=False)
    
    

    