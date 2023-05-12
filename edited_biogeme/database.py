import os
from expressions import *
import exceptions as excep
from datetime import datetime
import pandas as pd
import logging
import filenames as bf

class Database:
    def __init__(self,name,pandasDatabase):
        start_time = datetime.now()
        self.name = name
        self.data = pandasDatabase
        self.dataProcessingTime = datetime.now() - start_time
        self.generateHeaders()
        self.excludedData = 0
        self.panelColumn = None
        self.individualMap = None
        self.randomNumberGenerators = dict()
        listOfErrors,listOfWarnings = self.audit()
        if listOfWarnings:
            logging.warning('\n'.join(listOfWarnings))
        if listOfErrors:
            logging.error('\n'.join(listOfErrors))
            raise excep.biogemeError("\n".join(listOfErrors))
        
    def containsOnlyNumbers(df):
        return all(np.issubdtype(dtype, np.number) for dtype in df.dtypes)

    def audit(self):
        listOfErrors = []
        listOfWarnings = []
        for col,dtype in self.data.dtypes.items():
            if not np.issubdtype(dtype, np.number):
                theError = "Column {} in the database does contain {}".format(col,dtype)
                listOfErrors.append(theError)

        
        for cols in self.data.columns:
            if self.data[cols].isnull().any():
                theError = "Column {} in the database contains {} NaN value(s)".format(cols,self.data[cols].isnull().sum())
                listOfErrors.append(theError)
                
            try:
                self.data[cols].apply(pd.to_numeric)
            except ValueError:
                theError = "Column {} in the database contains strings".format(cols)
                listOfErrors.append(theError)
        return listOfErrors,listOfWarnings
            
    def generateHeaders(self):
        file = open("headers.py","w")
        print("from biogeme.expressions import *",file=file)
        for c in self.data.columns:
            globals()[c] = Variable(c)
            print("{}=Variable('{}')".format(c,c),file=file)
        file.close()

    def valuesFromDatabase(self,expression):
        self.expression = expression
        def functionToApply(row):
            self.expression.setRow(row)
            res = self.expression.getValue()
            return res
        res =  self.data.apply(functionToApply,axis=1)
        return res

    def checkAvailabilityOfChosenAlt(self,avail,choice):
        self.avail = avail
        self.choice = choice
        def functionToApply(row):
            self.choice.setRow(row)
            chosen = self.choice.getValue()
            avExpression = self.avail[chosen]
            avExpression.setRow(row)
            av = avExpression.getValue()
            return av != 0
        res = self.data.apply(functionToApply,axis=1)
        return res
            
    def sumFromDatabase(self,expression):
        self.expression = expression
        def functionToApply(row):
            self.expression.setRow(row)
            res = self.expression.getValue()
            return res
        res =  np.nansum(self.data.apply(functionToApply,axis=1))
        return res

    def sampleWithReplacement(self,size=None):
        if size is None:
            size = len(self.data)
        sample = self.data.iloc[np.random.randint(0, len(self.data), size=size)]
        return sample

    def sampleIndividualMapWithReplacement(self,size=None):
        if size is None:
            size = len(self.individualMap)
        sample = self.individualMap.iloc[np.random.randint(0, len(self.individualMap), size=size)]
        return sample
    
    def valueAndGradientFromDatabase(self,expression,literals,hessian=False):
        self.expression = expression
        def functionToApply(row):
            self.expression.setRow(row)
            res = self.expression.getValueAndGradient(literals,hessian)
            return res
        details =  self.data.apply(functionToApply,axis=1)
        if hessian:
            res = np.nansum(np.array([f for (f,g,h) in details])),np.nansum(np.array([g for (f,g,h) in details]),axis=0),np.nansum(np.array([h for (f,g,h) in details]),axis=0)
        else:
            res = np.nansum(np.array([f for (f,g) in details])),np.nansum(np.array([g for (f,g) in details]),axis=0)
        return res


    def bhhh(self,expression,literals):
        self.expression = expression
        def functionToApply(row):
            self.expression.setRow(row)
            res = self.expression.getValueAndGradient(literals,False)
            return res
        details =  self.data.apply(functionToApply,axis=1)
        res = np.nansum(np.array([np.outer(g,g) for (f,g) in details]),0)
        return res
    
    def applyExpression(self,expression):
        def functionToApply(row):
            self.expression.setRow(row)
            return self.expression.getValue()
        self.expression = expression
        return self.data.apply(functionToApply,axis=1)
        
    def addColumn(self,expression,column):
        if column in self.data.columns:
            raise ValueError(f"Column {column} already exists in the database {self.name}")

        def functionToApply(row):
            self.expression.setRow(row)
            return self.expression.getValue()
        self.expression = expression
        self.data[column] = self.data.apply(functionToApply,axis=1)

    def remove(self,expression):
        columnName = '__bioRemove__'
        if isNumeric(expression):
            self.addColumn(Numeric(expression),columnName)
        else:
            self.addColumn(expression,columnName)
        self.excludedData = len(self.data[self.data[columnName] != 0].index)
        self.data.drop(self.data[self.data[columnName] != 0].index, inplace=True)
        self.data.drop(columns=[columnName],inplace=True)

    def dumpOnFile(self):
        theName = "{}_dumped".format(self.name)
        dataFileName = bf.getNewFileName(theName,"dat")
        self.data.to_csv(dataFileName,sep='\t',index_label="__rowId")
        logging.info("File {} has been created".format(dataFileName))

    def setRandomNumberGenerators(self,rng):
        if 'NORMAL' in rng:
            raise ValueError("'NORMAL' is a reserved keyword for draws and cannot be used for user-defined generators") ;
        if 'UNIFORM' in rng:
            raise ValueError("'UNIFORM' is a reserved keyword for draws and cannot be used for user-defined generators") ;
        if 'UNIFORMSYM' in rng:
            raise ValueError("'UNIFORMSYM' is a reserved keyword for draws and cannot be used for user-defined generators") ;

        self.randomNumberGenerators = rng ;
    
    def generateDraws(self,types,names,numberOfDraws):
        self.numberOfDraws = numberOfDraws
        # Dimensions of the draw table:
        # 1. number of variables
        # 2. number of individuals
        # 3. number of draws
        listOfDraws = [None]*len(names)
        for i in range(len(names)):
            name = names[i]
            type = types[name]
            if type == "NORMAL":
                listOfDraws[i] = np.random.randn(self.getSampleSize(),numberOfDraws)
            elif type == "UNIFORM":
                listOfDraws[i] = np.random.rand(self.getSampleSize(),numberOfDraws)
            elif type == "UNIFORMSYM":
                listOfDraws[i] = 2.0 * np.random.rand(self.getSampleSize(),numberOfDraws) - 1.0
            else:
                theGenerator = self.randomNumberGenerators.get(type)
                if theGenerator is None:
                    raise biogemeError("Unkown type of draws for variable {}: {}".format(name,type))
                else:
                    listOfDraws[i] = theGenerator((self.getSampleSize(),numberOfDraws))
                    
        self.theDraws = np.array(listOfDraws)
        # We reorganize the dimensions to obtain an organization more
        # suited for calculation. 
        # 1. number of individuals
        # 2. number of draws
        # 3. number of variables
        self.theDraws = np.moveaxis(self.theDraws,0,-1)

    def getNumberOfObservations(self):
        return self.data.shape[0]
            
    def getSampleSize(self):
        if self.isPanel():
            return self.individualMap.shape[0]
        else:
            return self.data.shape[0]


    def isPanel(self):
        return self.panelColumn is not None
    
    def panel(self,columnName):
        self.panelColumn = columnName
    
    def buildPanelMap(self):
        if self.panelColumn is not None:
            self.data = self.data.sort_values(by=self.panelColumn)
            # It is necessary to renumber the row to reflect the new ordering
            self.data.index = range(len(self.data.index))        
            map = dict()
            individuals = self.data[self.panelColumn].unique()
            for i in individuals:
                indices = self.data.loc[self.data[self.panelColumn] == i].index
                map[i] = [min(indices), max(indices)]
            self.individualMap = pd.DataFrame(map).T

    def count(self,columnName,value):
        return self.data[self.data[columnName] == value].count()[columnName]
