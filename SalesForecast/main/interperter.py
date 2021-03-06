'''
Created on 21 Aug 2018

@author: seanv
'''

#warnings
import warnings
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
from pandas import Series
print('pandas: %s' % pandas.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)

from main import variables

#Data
from Data import validate
#validate.split_datafile(variables.datafileloc, variables.datasetloc, variables.validationloc)

#Model evaluation
from Model import performence

#series = Series.from_csv(variables.datasetloc)
#p_values = range(0, 7)
##d_values = range(0, 3)
#q_values = range(0, 7)

#best_p = performence.best_p
#best_d = performence.best_d
#best_q = performence.best_q
#warnings.filterwarnings("ignore")

'''
 Step one: Eveluate the models, use the model with the lowest RMSE
 
 Uncomment 'performence.evaluate_models(series.values, p_values, d_values, q_values)'
 
 Next uncomment 'performence.checkbias(0, 0, 1)' and replace the (0,0,1) with the best RSME values
 
 copy and paste the value of 'mean'
'''
#performence.evaluate_models(series.values, p_values, d_values, q_values)


#best_p, best_d, best_q = (performence.getBestcfg())
#performence.checkbias(best_p, best_d, best_q)


#print(performence.getBias())
#best_bias = performence.getBias()

'''
 Step two: Creating the ARIMA-model
 
 uncomment '#arima.arimaModel(0,0,1,165.904727)' and add the best RSME values and the value of mean
 
 
'''
#Model
from Model import test
from Model import arima
#test.test_function()
#arima.arimaModel(best_p,best_d,best_q, best_bias)

#Prediction
from Model import predict

#predict.predictFuture() #predicts beyond datafile without validation through training
#predict.predictRun(best_p,best_d,best_q) #predicts next values of dataset with validation from datafile values


def newDataserRun():
    validate.split_datafile(variables.datafileloc, variables.datasetloc, variables.validationloc)
    
    series = Series.from_csv(variables.datasetloc)
    p_values = range(0, 7)
    d_values = range(0, 3)
    q_values = range(0, 7)
    
    warnings.filterwarnings("ignore")
    
    performence.evaluate_models(series.values, p_values, d_values, q_values)


    best_p, best_d, best_q = (performence.getBestcfg())
    performence.checkbias(best_p, best_d, best_q)
    #print(performence.getBias())
    best_bias = performence.getBias()
    
    arima.arimaModel(best_p,best_d,best_q, best_bias)
    
    predict.predictRun(best_p,best_d,best_q)



    
def loadDatasetRun():
    validate.split_datafile(variables.datafileloc, variables.datasetloc, variables.validationloc)
    predict.predictFuture() 
    
#newDataserRun()
loadDatasetRun()
#Analysis
from Analysis import Summary
from Analysis import LinePlot
from Analysis import SeasonalLine
from Analysis import DensityPlot
from Analysis import BoxAndWhiskerPlot
#Summary.getSummary()
#LinePlot.getLineplot()
#SeasonalLine.getSeasonalLine() #BROKEN
#DensityPlot.getDensityPlot()
#BoxAndWhiskerPlot.getBoxWhiskerPlot() #BROKEN

