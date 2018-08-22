'''
Created on 21 Aug 2018

@author: seanv
'''
def getSummary():
    from Data import validate

    file = validate.getdatafile()
    summary = file.describe()
    print(summary)
