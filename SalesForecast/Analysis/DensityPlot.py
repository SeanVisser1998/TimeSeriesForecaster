'''
Created on 21 Aug 2018

@author: seanv
'''
from Data import validate
from matplotlib import pyplot

def getDensityPlot():
    X = validate.getdatafile()
    X.astype('float32')
    pyplot.figure(1)
    pyplot.subplot(211)
    X.hist()
    pyplot.subplot(212)
    X.plot(kind='kde')
    pyplot.show()