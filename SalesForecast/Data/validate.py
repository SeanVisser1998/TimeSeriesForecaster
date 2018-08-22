'''
Created on 21 Aug 2018

@author: seanv
'''
from pandas import Series

def split_datafile(filelocation, datasetlocation, validationlocation):
    
    split_datafile.file = Series.from_csv(filelocation)
    
    split_point = len(split_datafile.file) - 12
    split_datafile.dataset, split_datafile.validation = split_datafile.file[0:split_point], split_datafile.file[split_point:]
    split_datafile.dataset.to_csv(datasetlocation)
    split_datafile.validation.to_csv(validationlocation)
    print('Dataset %d, Validation %d' % (len(split_datafile.dataset), len(split_datafile.validation)))
   
def getdataset():
    return split_datafile.dataset  

def getvalidation():
    return split_datafile.validation

def getdatafile():
    return split_datafile.file
    