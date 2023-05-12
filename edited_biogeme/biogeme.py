"""
Main class that combines the database and the model specification. It works 
in two modes: estimation and simulation.
"""
import numpy as np
import pandas as pd
import pickle 
import logging
from datetime import datetime
from scipy.optimize import minimize
import multiprocessing as mp

import database as db
import cbiogeme as cb
import expressions as eb
import tools as tools
import version as bv
import results as res
import exceptions as excep
import filenames as bf

import time
#import yep

#Default formatting of the logging 
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt='%d-%m-%Y:%H:%M:%S',level=logging.INFO)



class BIOGEME:
    """Main class that combines the database and the model specification. It works 
    in two modes: estimation and simulation. 

    Args:
        :param database: choice data
        :type database: biogeme.database

        :param formulas: expression or dictionary of expressions that define 
             the model specification.  The concept is that each expression is 
             applied to each entry of the database. The keys of the dictionary
             allow to provide a name to each formula.
 
             In the estimation mode, two formulas are needed, with the 
             following keys: 'loglike' and 'weight'. If only one formula 
             is provided, it is associated with the label 'loglike'. If no
             formula is labeled 'weight', the weight of each piece of data
             is supposed to be 1.0. 

             In the simulation mode, the labels of each formula are used as
             labels of the resulting database. 
        :type formulas: [biogeme.expressions, dict(biogeme.expressions)]

        :param numberOfThreads: multi-threading can be used for
            estimation. This parameter defines the number of threads
            to be used. If the parameter is set to None, the number of
            available threads is calculated using
            cpu_count(). Ignored in simulation mode. Default: None
        :type numberOfThreads: int

        :param numberOfDraws: number of draws used for Monte-Carlo
            integration. Default: 1000
        :type numberOfDraws: int

        :param seed: seed used for the pseudo-random number
            generation. It is useful only when each run should
            generate the exact same result. If None, a new seed is
            used at each run. Default: None
        :type seed: int
    """
    def __init__(self,database,formulas,numberOfThreads=None,numberOfDraws=1000,seed=None):
        database.data = database.data.replace({True:1,False:0})
        listOfErrors,listOfWarnings = database.audit()
        if listOfWarnings:
            logging.warning('\n'.join(listOfWarnings))
        if listOfErrors:
            logging.error('\n'.join(listOfErrors))
            raise excep.biogemeError("\n".join(listOfErrors))
        
        self.loglikeName = 'loglike'
        self.weightName = 'weight'
        self.modelName = 'biogemeModelDefaultName'
        # monteCarlo is True if one of the expression invovles a
        # Monte-Carlo integration
        self.monteCarlo = False
        np.random.seed(seed)

        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG)
        
        if type(formulas) is not dict:
            self.loglike = formulas
            self.weight = None
            self.formulas = dict({self.loglikeName:formulas})
        else:
            self.loglike = formulas.get(self.loglikeName)
            self.weight = formulas.get(self.weightName)
            self.formulas = formulas
            
        self.database = database
        self.audit()
        if self.database.isPanel():
            self.database.buildPanelMap()
            self.theC = cb.pyPanelBiogeme()
            self.theC.setDataMap(self.database.individualMap)
        else:
            self.theC = cb.pyBiogeme()
        self.theC.setData(self.database.data)
        self._prepareLiterals()
        self.generateHtml = True
        self.numberOfThreads = mp.cpu_count() if numberOfThreads is None else numberOfThreads
        start_time = datetime.now()        
        self._generateDraws(numberOfDraws)
        if self.monteCarlo:
            self.theC.setDraws(self.database.theDraws)
        self.drawsProcessingTime = datetime.now() - start_time
        if self.loglike is not None:
            self.loglikeSignatures = self.loglike.getSignature()

            if self.weight is None:
                self.theC.setExpressions(self.loglikeSignatures,self.numberOfThreads)
            else:
                self.weightSignatures = self.weight.getSignature()
                self.theC.setExpressions(self.loglikeSignatures,self.numberOfThreads,self.weightSignatures)

    
    def audit(self):
        """Each expression provides an audit function, that verifies it
           validity. Each formula is audited, and the list of errors
           and warnings reported.

           Args: None
          
           Return: nothing

           Raise: biogemeError if an error is detected.

        """
        listOfErrors = []
        listOfWarnings = []
        for k,v in self.formulas.items() :
            err, war = v.audit(self.database)
            listOfErrors += err
            listOfWarnings += war
        if listOfWarnings:
            logging.warning('\n'.join(listOfWarnings))
        if listOfErrors:
            logging.error('\n'.join(listOfErrors))
            raise excep.biogemeError("\n".join(listOfErrors))
        
    def _generateDraws(self,numberOfDraws):
        """If Monte-Carlo integration is involved in one of the formulas, this
           function instructs the database to generate the draws.

        Args:
            :param numberOfDraws: self explanatory
            :type numberOfDraws: int
        """
        
        self.numberOfDraws = numberOfDraws
        drawTypes = {}
        for k,v in self.formulas.items() :
            d = v.getDraws()
            if d:
                drawTypes = dict(drawTypes,**d)
        self.monteCarlo = bool(drawTypes)
        if self.monteCarlo:
            self.monteCarloNames = sorted(drawTypes)
            self.monteCarloIds = []
            for i in range(len(self.monteCarloNames)):
                for k,v in self.formulas.items() :
                    v.setDrawIndex(self.monteCarloNames[i],i)
                self.monteCarloIds.append(i)
            self.database.generateDraws(drawTypes,
                                        self.monteCarloNames,
                                        numberOfDraws)
            
    def _prepareLiterals(self):
        """ Extract from the formulas the literals (parameters,
        variables, random variables) and decide a numbering convention.
        """ 
        self.bounds = list()
        allBetas = dict()
        for k,v in self.formulas.items() :
            d = v.dictOfBetas(free=True,fixed=False)
            allBetas = dict(allBetas,**d)
        self.freeBetaNames = sorted(allBetas)
        for x in self.freeBetaNames:
            self.bounds.append((allBetas[x].lb,allBetas[x].ub))
        self.betaIds = []
        for i in range(len(self.freeBetaNames)):
            for k,v in self.formulas.items() :
                v.setIndex(self.freeBetaNames[i],i)
            self.betaIds.append(i)

        self.betaInitValues = list() ;
        for x in self.freeBetaNames:
            self.betaInitValues.append(float(allBetas[x].initValue))
        allFixedBetas = dict()
        for k,v in self.formulas.items() :
            d = v.dictOfBetas(free=False,fixed=True)
            allFixedBetas = dict(allFixedBetas,**d)
        self.fixedBetaNames = sorted(allFixedBetas)
        for i in range(len(self.fixedBetaNames)):
            for k,v in self.formulas.items() :
                v.setIndex(self.fixedBetaNames[i],i+len(self.freeBetaNames))
        self.fixedBetaValues = list() ;
        for x in self.fixedBetaNames:
            self.fixedBetaValues.append(float(allFixedBetas[x].initValue))
            
        variableNames = list(self.database.data.columns.values)
        for i in range(len(variableNames)):
            for k,v in self.formulas.items() :
                index = i+len(self.freeBetaNames)+len(self.fixedBetaNames)
                v.setIndex(variableNames[i],index)

        randomVariables = dict()
        for k,v in self.formulas.items() :
            d = v.dictOfRandomVariables()
            randomVariables = dict(randomVariables,**d)
        self.rvNames = sorted(randomVariables)
        for i in range(len(self.rvNames)):
            for k,v in self.formulas.items() :
                v.setIndex(self.rvNames[i],i)
        
    
    def calculateInitLikelihood(self):
        """Calculate the value of the log likelihood function using the
           default values of the parameters.

           Return: value of the log likelihood (real)
        """
        self.initLogLike = self.calculateLikelihood(self.betaInitValues)
        return self.initLogLike

    def calculateLikelihood(self,x):
        """ Calculate the value of the log likelihood function 

        Args:
            :param x: vector of values for the parameters
            :type x: list(float)

        Returns:
            the calculated value (float)

        Raise:
            ValueError if the length of the list x is incorrect

        """
        if len(x) != len(self.betaInitValues):
            raise ValueError("Input vector must be of length {} and not {}".format(len(self.betaInitValues),len(x)))

        f = self.theC.calculateLikelihood(x,self.fixedBetaValues)
        return f

    def calculateLikelihoodAndDerivatives(self,x,hessian=False,bhhh=False):
        """Calculate the value of the log likelihood function and its
           derivatives.

        Args:
            :param x: vector of values for the parameters
            :type x: list(float)
            :param hessian: if True, the hessian is calculated
            :type hessian: bool
            :param bhhh: if Truem the BHHH matrix is calculated
            :type bhhh: bool

        Returns:
            a tuple f,g,h,bh where
                - f (float) is the value of the function
                - g (numpy.array) is the gradient
                - h (numpy.array) is the hessian
                - bh (numpy.array) is the BHHH matrix
        Raise:
            ValueError if the length of the list x is incorrect

        """
        if len(x) != len(self.betaInitValues):
            raise ValueError("Input vector must be of length {} and not {}".format(len(betaInitValues),len(x)))
        f,g,h,bh = self.theC.calculateLikelihoodAndDerivatives(x,
                                                               self.fixedBetaValues,
                                                               self.betaIds,
                                                               hessian,
                                                               bhhh)

        return f,np.asarray(g),np.asarray(h), np.asarray(bh)

    def likelihoodFiniteDifferenceHessian(self,x):
        """Calculate the hessian of the log likelihood function using finite
           differences. May be useful when the analytical hessian has
           numerical issues. 

        Args:
            :param x: vector of values for the parameters
            :type x: list(float)

        Returns:
            A numpy.array with the finite differences approximation of
            the hessian

        Raise:
            ValueError if the length of the list x is incorrect

        """
        def theFunction(x):
            f,g,h,b = self.calculateLikelihoodAndDerivatives(x,hessian=False,bhhh=False)
            return f,np.asarray(g)
        return tools.findiff_H(theFunction,x)

    def checkDerivatives(self,verbose=False):
        """Verifies the implementation of the derivatives by comparing the
           analytical version with the finite differences
           approximation.

        Args:
            :param verbose: if True, the comparisons are
                reported. Default: False.
            :type verbose: bool

        Returns: tuple f,g,h,gdiff,hdiff where
            - f is the value of the function
            - g is the analytical gradient
            - h is the analytical hessian
            - gdiff is the difference between the analytical and the
              finite differences gradient
            - hdiff is the difference between the analytical and the
              finite differences hessian
        """
        def theFunction(x):
            f,g,h,b = self.calculateLikelihoodAndDerivatives(x,hessian=True,bhhh=False)
            return f,np.asarray(g),np.asarray(h)

        return tools.checkDerivatives(theFunction,np.asarray(self.betaInitValues),self.freeBetaNames,verbose)

    def estimate_baichuan(self, max_iter = None, method = 'BGFS', Write_html = False):
        """Estimate the parameters of the model.

        Args:
            :param bootstrap: number of bootstrap resampling used to
               calculate the variance-covariance matrix using
               bootstrapping. If the number is 0, bootstrapping is not
               applied. Default: 0.
            :type bootstrap: int

        Returns biogeme.bioResults object containing the estimation results.

        Raises:
            biogemeError if no expression has been provided for the likelihood

        """
        if self.loglike is None:
            raise excep.biogemeError("No log likelihood function has been specificed")

        self.calculateInitLikelihood()

        def f_and_grad(x):
            f, g, h, b = self.calculateLikelihoodAndDerivatives(x, hessian=False, bhhh=False)
            scale = np.abs(self.initLogLike) * 100.0
            return -f / scale, -np.asarray(g / scale)

        start_time = datetime.now()
        #        yep.start('profile.out')

        opts = {'maxiter': max_iter,'disp':True}
        print('minimize start') #Newton-CG
        results = minimize(f_and_grad, self.betaInitValues, method=method, jac=True, options=opts)
        # results = minimize(f_and_grad, self.betaInitValues, bounds=self.bounds, jac=True,
        #                    options=opts)
        #        yep.stop()
        print('minimize finished')

        if Write_html:
            print('start write html')
            s_time = time.time()
            self.optimizationTime = datetime.now() - start_time
            self.optimizationMessage = results.message
            self.numberOfFunctionEval = results.nfev
            self.numberOfIterations = results.nit
            fgHb = self.calculateLikelihoodAndDerivatives(results.x, hessian=True, bhhh=True)
            rawResults = res.rawResults(self,results.x,fgHb,bootstrap=None)
            r = res.bioResults(rawResults)
            r.writeHtml()
            print('write html finished')
            write_html_time = time.time() - s_time # second
        else:
            write_html_time = 0
        # opts = {'gtol' : 1e-7,'ftol' : np.finfo(float).eps}
        # results =  minimize(f_and_grad,self.betaInitValues,bounds=self.bounds,jac=True,options=opts)
        # #        yep.stop()

        return results.x, self.freeBetaNames, write_html_time

    def estimate(self,bootstrap=0):
        """Estimate the parameters of the model. 

        Args:
            :param bootstrap: number of bootstrap resampling used to
               calculate the variance-covariance matrix using
               bootstrapping. If the number is 0, bootstrapping is not
               applied. Default: 0.
            :type bootstrap: int

        Returns biogeme.bioResults object containing the estimation results. 

        Raises:
            biogemeError if no expression has been provided for the likelihood

        """
        if self.loglike is None:
            raise excep.biogemeError("No log likelihood function has been specificed")
            
        self.calculateInitLikelihood()

        def f_and_grad(x):
            f,g,h,b = self.calculateLikelihoodAndDerivatives(x,hessian=False,bhhh=False)
            scale = np.abs(self.initLogLike) * 100.0
            return -f/scale,-np.asarray(g/scale)

        start_time = datetime.now()
        #        yep.start('profile.out')

        opts = {'maxiter': 10}
        # print('minimize start')
        # results =  minimize(f_and_grad,self.betaInitValues,method='BFGS',jac=True,options=opts)
        # results = minimize(f_and_grad, self.betaInitValues, bounds=self.bounds, jac=True,
        #                    options=opts)
        #        yep.stop()
        # print ('minimize finished')

        opts = {'gtol' : 1e-7,'ftol' : np.finfo(float).eps}
        results =  minimize(f_and_grad,self.betaInitValues,bounds=self.bounds,jac=True,options=opts)
        #        yep.stop()

        self.optimizationTime = datetime.now() - start_time
        self.optimizationMessage = results.message
        self.numberOfFunctionEval = results.nfev
        self.numberOfIterations = results.nit
        fgHb = self.calculateLikelihoodAndDerivatives(results.x,hessian=True,bhhh=True)
        if not np.isfinite(fgHb[2]).all():
            logging.warning("Numerical problems in calculating the analytical hessian. Finite differences is tried instead.")
            finDiffHessian = self.likelihoodFiniteDifferenceHessian(results.x)
            if not np.isfinite(fgHb[2]).all():
                logging.warning("Numerical problems with finite difference hessian as well.")
            else:
                fgHb = fgHb[0],fgHb[1],finDiffHessian,fgHb[3]
        self.bootstrapResults = None
        if bootstrap > 0:
            start_time = datetime.now()

            logging.info("Re-estimate the model {} times for bootstrapping".format(bootstrap))
            self.bootstrapResults = np.empty(shape=[bootstrap, len(results.x)])
            for b in range(bootstrap):
                print(".",end='',flush=True)
                if self.database.isPanel():
                    sample = self.database.sampleIndividualMapWithReplacement()
                    self.theC.setDataMap(sample)
                else:
                    sample = self.database.sampleWithReplacement()
                    self.theC.setData(sample)
                br = minimize(f_and_grad,results.x,bounds=self.bounds,jac=True,options={'ftol':1.0e-10,'gtol':1.0e-5})
                self.bootstrapResults[b] = br.x
            print("*")
            
            self.bootstrapTime = datetime.now() - start_time

        rawResults = res.rawResults(self,results.x,fgHb,bootstrap=self.bootstrapResults)
        r = res.bioResults(rawResults)
        if self.generateHtml:
            r.writeHtml()
        return r
    
    def simulate(self,theBetaValues=None):
        """Applies the formulas to each row of the database.
        
        Args:
            :param theBetaValues: values of the parameters to be used in
                the calculations. If None, the default values are
                used. Default: None.  
            :type theBetaValues: dict(str,float) or list(float)
          
        Returns a pandas data frame with the simulated value. Each row
        corresponds to a row in the database, and each column to a
        formula.

        Raise:
            biogemeError: if the number of parameters is incorrect
        """

        if theBetaValues is None:
            betaValues = self.betaInitValues
        else:
            if type(theBetaValues) is not dict:
                if len(theBetaValues) != len(self.betaInitValues):
                    err = "The value of {} parameters should be provided, not {}".format(len(self.betaInitValues),len(theBetaValues))
                    raise excep.biogemeError(err)
                
                betaValues = theBetaValues
            else:
                betaValues = list()
                for i in range(len(self.freeBetaNames)):
                    x = self.freeBetaNames[i]
                    if x in theBetaValues:
                        betaValues.append(theBetaValues[x])
                    else:
                        betaValues.append(self.betaInitValues[i])
                            
        output = pd.DataFrame(index=self.database.data.index)
        for k,v in self.formulas.items():
            signature = v.getSignature()
            result = self.theC.simulateFormula(signature,betaValues,self.fixedBetaValues,self.database.data)
            output[k] = result
        return output

    
    def confidenceIntervals(self,betaValues,intervalSize=0.9):
        """Calculate confidence intervals on the simulated quantities

        Args: 
            :param betaValues: array of parameters values to be used
               in the calculations. Typically, it is a sample drawn from
               a distribution.  
            :type betaValues: list(dict(float))

            :param intervalSize: size of the reported confidence
                    interval, in percentage. If it is denoted by s,   
                    the interval is calculated for the quantiles (1-s)/2 and 
                    (1+s)/2. Default: 0.9. Associated quantiles for the 
                    confidence interval: 0.05 and 0.95

        Returns: two pandas data frames 'left' and 'right' with the same 
                 dimensions. 
                 Each row corresponds to a row in the database, and each 
                 column to a formula. 'left' contains the left value of
                 the confidence interval, and 'right' the right value
        """
        listOfResults = []
        for b in betaValues:
            r = self.simulate(b)
            listOfResults += [r]
        allResults = pd.concat(listOfResults)
        r = (1.0-intervalSize)/2.0
        left = allResults.groupby(level=0).quantile(r)
        right = allResults.groupby(level=0).quantile(1.0-r)
        return left,right
