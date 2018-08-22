'''
Created on 21 Aug 2018

@author: seanv
'''
from Data import validate
from matplotlib import pyplot
def getLineplot():
    
    X = validate.getdatafile()
    X.plot()
    pyplot.title('Lijn diagram')
    pyplot.show()