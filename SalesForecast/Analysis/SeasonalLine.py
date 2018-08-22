'''
Created on 21 Aug 2018

@author: seanv

BROKEN
'''

from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
from Data import validate

def getSeasonalLine():
    X = validate.getdataset()
    X.astype('float32')
    groepen = X['1964':'1970'].groupby(TimeGrouper('A'))
    jaren = DataFrame()
    pyplot.figure()
    i = 1
    n_groep = len(groepen)
    for name, groep in groepen:
        pyplot.subplot((n_groep * 100) + 10 + i)
        i += 1 
        pyplot.plot(groep)
    pyplot.show()
        
    
