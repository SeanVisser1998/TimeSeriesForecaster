'''
Created on 21 Aug 2018

@author: seanv

BROKEN
'''
from Data import validate
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot

def getBoxWhiskerPlot():
    X = validate.getdatafile()
    X.astype('float32')
    groepen = X['1964':'1970'].groupby(TimeGrouper('A'))
    jaren = DataFrame()
    for name, groep in groepen:
       jaren[name.year] = groep.values
    jaren.boxplot()
    pyplot.show()