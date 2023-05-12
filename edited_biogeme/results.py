import pickle
import datetime
import logging
import pandas as pd
import numpy as np
from scipy import linalg
from scipy import stats
import version as bv
import filenames as bf
import exceptions as excep


def calcPValue(t):
    """ Calculates the p value of a parameter from its t-statistic.
    Args:
        :param t: t-statistics
        :type real
    Return: p-value
    """
    p = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    return p

class beta:
    """ Class gathering the information related to the parameters of the model

    Args:
        :param name: name of the parameter
        :type str

        :param: value: value of the parameter
        :type float
    """
    def __init__(self,name,value):
        self.name = name
        self.value = value
        self.stdErr = None
        self.tTest = None
        self.pValue = None
        self.robust_stdErr = None
        self.robust_tTest = None
        self.robust_pValue = None
        self.bootstrap_stdErr = None
        self.bootstrap_tTest = None
        self.bootstrap_pValue = None
        

    def setStdErr(self,se):
        """ Records the standard error, and calculates and records 
            the corresponding t-statistic and p-value
        Args:
            :param se: standard error

        Returns: nothing
        """
        self.stdErr = se
        self.tTest = np.nan_to_num(self.value / se)
        self.pValue = calcPValue(self.tTest)

    def setRobustStdErr(self,se):
        """ Records the robust standard error, and calculates and records 
            the corresponding t-statistic and p-value
        Args:
            :param se: robust standard error
        
        Returns: nothing
        """
        self.robust_stdErr = se
        self.robust_tTest = np.nan_to_num(self.value / se)
        self.robust_pValue = calcPValue(self.robust_tTest)

    def setBootstrapStdErr(self,se):
        self.bootstrap_stdErr = se
        self.bootstrap_tTest = np.nan_to_num(self.value / se)
        self.bootstrap_pValue = calcPValue(self.robust_tTest)
        
        
    def __str__(self):
        str = "{:15}: {:.3g}".format(self.name,self.value)
        if self.stdErr is not None:
            str += "[{:.3g} {:.3g} {:.3g}]".format(self.stdErr,self.tTest,self.pValue)
        if self.robust_stdErr is not None:
            str += "[{:.3g} {:.3g} {:.3g}]".format(self.robust_stdErr,self.robust_tTest,self.robust_pValue)
        if self.bootstrap_stdErr is not None:
            str += "[{:.3g} {:.3g} {:.3g}]".format(self.bootstrap_stdErr,self.bootstrap_tTest,self.bootstrap_pValue)
        return str

class rawResults:
    def __init__(self,theModel,betaValues,fgHb,bootstrap=None):
        """ Ctor """

        self.modelName = theModel.modelName
        self.nparam = len(betaValues)
        self.betaValues = betaValues
        self.betaNames = theModel.freeBetaNames
        self.initLogLike = theModel.initLogLike
        self.betas = list()
        for b,n in zip(betaValues,self.betaNames):
            self.betas.append(beta(n,b))
        self.logLike = fgHb[0]
        self.g = fgHb[1]
        self.H = fgHb[2]
        self.bhhh = fgHb[3]
        self.dataname = theModel.database.name
        self.sampleSize = theModel.database.getSampleSize()
        self.numberOfObservations = theModel.database.getNumberOfObservations()
        self.monteCarlo = theModel.monteCarlo
        self.numberOfDraws = theModel.numberOfDraws
        self.excludedData = theModel.database.excludedData
        self.dataProcessingTime = theModel.database.dataProcessingTime
        self.drawsProcessingTime = theModel.drawsProcessingTime
        self.optimizationTime = theModel.optimizationTime
        self.gradientNorm = linalg.norm(self.g)
        self.optimizationMessage = theModel.optimizationMessage
        self.numberOfFunctionEval = theModel.numberOfFunctionEval
        self.numberOfIterations = theModel.numberOfIterations
        self.numberOfThreads = theModel.numberOfThreads
        self.htmlFileName = None
        self.latexFileName = None
        self.pickleFileName = None
        self.bootstrap = bootstrap
        if bootstrap is not None:
            self.bootstrapTime = theModel.bootstrapTime
    
class bioResults:
    def __init__(self,rawResults=None,pickleFile=None):
        if rawResults is not None:
            self.data = rawResults
            self.dumpData()
        elif pickleFile is not None:
            with open(pickleFile, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise excep.biogemeError("No data provided.")
            
        self.calculateStats()

    def dumpData(self): 
        self.data.pickleFileName = bf.getNewFileName(self.data.modelName,"pickle")
        with open(self.data.pickleFileName, 'wb') as f:
            pickle.dump(self.data, f)
#        print("Results dumped on file {}".format(pickleFile))
#        print("Can be read using the following statement:")
#        print("import pickle") 
#        print("with open('{}', 'rb') as f:".format(pickleFile))
#        print("    data = pickle.load(f)")
        
    def calculateTest(self,i,j,matrix):
        vi = self.data.betaValues[i]
        vj = self.data.betaValues[j]
        varI = matrix[i,i]
        varJ = matrix[j,j]
        covar = matrix[i,j]
        test = np.nan_to_num((vi - vj) / np.sqrt(varI + varJ - 2.0 * covar))
        return test
    
    def calculateStats(self):
        self.data.likelihoodRatioTest = -2.0 * (self.data.initLogLike - self.data.logLike)
        self.data.rhoSquare = np.nan_to_num(1.0 - self.data.logLike / self.data.initLogLike)
        self.data.rhoBarSquare = np.nan_to_num(1.0 - (self.data.logLike-self.data.nparam) / self.data.initLogLike)
        self.data.akaike = 2.0 * self.data.nparam - 2.0 * self.data.logLike
        self.data.bayesian = - 2.0 * self.data.logLike + self.data.nparam * np.log(self.data.sampleSize) 
        # We calculate the eigenstructure to report in case of singularity
        self.data.eigenValues,self.data.eigenvectors = linalg.eigh(-np.nan_to_num(self.data.H))
        U, self.data.singularValues,V = linalg.svd(-np.nan_to_num(self.data.H))
        # We use the pseudo inverse in case the matrix is singular
        self.data.varCovar = -linalg.pinv(np.nan_to_num(self.data.H))
        for i in range(self.data.nparam):
            self.data.betas[i].setStdErr(np.sqrt(self.data.varCovar[i,i]))

        d = np.diag(self.data.varCovar)
        if (d > 0).all():
            diag = np.diag(np.sqrt(d))
            diagInv = linalg.inv(diag)
            self.data.correlation = diagInv.dot(self.data.varCovar.dot(diagInv))
        else:
            self.data.correlation = np.full_like(self.data.varCovar,np.finfo(float).max)

        
        # Robust estimator
        self.data.robust_varCovar = self.data.varCovar.dot(self.data.bhhh.dot(self.data.varCovar))
        for i in range(self.data.nparam):
            self.data.betas[i].setRobustStdErr(np.sqrt(self.data.robust_varCovar[i,i]))
        rd = np.diag(self.data.robust_varCovar)
        if (rd > 0).all():
            diag = np.diag(np.sqrt(rd))
            diagInv = linalg.inv(diag)
            self.data.robust_correlation = diagInv.dot(self.data.robust_varCovar.dot(diagInv))
        else:
            self.data.robust_correlation = np.full_like(self.data.robust_varCovar,np.finfo(float).max)
            
        # Bootstrap
        if self.data.bootstrap is not None:
            self.data.bootstrap_varCovar = np.cov(self.data.bootstrap,rowvar=False)
            for i in range(self.data.nparam):
                self.data.betas[i].setBootstrapStdErr(np.sqrt(self.data.bootstrap_varCovar[i,i]))
            rd = np.diag(self.data.bootstrap_varCovar)
            if (rd > 0).all():
                diag = np.diag(np.sqrt(rd))
                diagInv = linalg.inv(diag)
                self.data.bootstrap_correlation = diagInv.dot(self.data.bootstrap_varCovar.dot(diagInv))
            else:
                self.data.bootstrap_correlation = np.full_like(self.data.bootstrap_varCovar,np.finfo(float))

        self.data.secondOrderTable = dict()
        for i in range(self.data.nparam):
            for j in range(i):
                t = self.calculateTest(i,j,self.data.varCovar)
                p = calcPValue(t)
                trob = self.calculateTest(i,j,self.data.robust_varCovar)
                prob = calcPValue(trob)
                if self.data.bootstrap is not None:
                    tboot = self.calculateTest(i,j,self.data.bootstrap_varCovar)
                    pboot = calcPValue(tboot)
                name = (self.data.betaNames[i],self.data.betaNames[j])
                if self.data.bootstrap is not None:
                    self.data.secondOrderTable[name] = [self.data.varCovar[i,j],self.data.correlation[i,j],t,p,self.data.robust_varCovar[i,j],self.data.robust_correlation[i,j],trob,prob,self.data.bootstrap_varCovar[i,j],self.data.bootstrap_correlation[i,j],tboot,pboot]
                else:
                    self.data.secondOrderTable[name] = [self.data.varCovar[i,j],self.data.correlation[i,j],t,p,self.data.robust_varCovar[i,j],self.data.robust_correlation[i,j],trob,prob]

        self.data.smallestEigenValue = min(self.data.eigenValues)
        self.data.smallestSingularValue = min(self.data.singularValues)

    def __str__(self):
        r = "\n"
        r += "Results for model ["+self.data.modelName+"]\n"
        if self.data.htmlFileName is not None :
            r += "Output file (HTML):\t\t\t{}\n".format(self.data.htmlFileName)
        if self.data.latexFileName is not None :
            r += "Output file (LaTeX):\t\t\t{}\n".format(self.data.latexFileName)
        r += "Nbr of parameters:\t\t{}\n".format(self.data.nparam)
        r += "Sample size:\t\t\t{}\n".format(self.data.sampleSize)
        if self.data.sampleSize != self.data.numberOfObservations:
            r += "Observations:\t\t\t{}\n".format(self.data.numberOfObservations)
        r += "Excluded data:\t\t\t{}\n".format(self.data.excludedData)
        r += "Init log likelihood:\t\t{:.7g}\n".format(self.data.initLogLike)
        r += "Final log likelihood:\t\t{:.7g}\n".format(self.data.logLike)
        r += "Likelihood ratio test:\t\t{:.7g}\n".format(self.data.likelihoodRatioTest)
        r += "Rho square:\t\t\t{:.3g}\n".format(self.data.rhoSquare)
        r += "Rho bar square:\t\t\t{:.3g}\n".format(self.data.rhoBarSquare)
        r += "Akaike Information Criterion:\t{:.7g}\n".format(self.data.akaike)
        r += "Bayesian Information Criterion:\t{:.7g}\n".format(self.data.bayesian)
        r += "Final gradient norm:\t\t{:.7g}\n".format(self.data.gradientNorm)
        r += "\n".join(["{}".format(b) for b in self.data.betas])
        r += "\n"
        for k,v in self.data.secondOrderTable.items():
            r += "{}:\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\n".format(k,*v)
        return r

    def getLaTeXHeader(self):
        h = ""
        h += "%% This file is designed to be included into a LaTeX document\n"
        h += "%% See http://www.latex-project.org/ for information about LaTeX\n"
        h += bv.getLaTeX()
        return h
    
    def getLaTeX(self):
        
        now = datetime.datetime.now()
        h = self.getLaTeXHeader()
        h += "\n%% File "+self.data.latexFileName
        h += "\n%% This file has automatically been generated on {}</p>\n".format(now)
        h += "\n%%Database name: "+self.data.dataname+"\n"
        h += "\n%% General statistics\n"
        h += "\\section{General statistics}\n"
        d = self.getGeneralStatistics()
        h += "\\begin{tabular}{ll}\n"
        for k,(v,p) in d.items():
            if isinstance(v,bytes):
                v = str(v)
            if isinstance(v, str):
                v = v.replace('_','\\_')
            h += f"{k} & {v:{p}} \\\\\n"
        h += "\\end{tabular}\n"

        h += "\n%%Parameter estimates\n"
        h += "\section{Parameter estimates}\n"
        table = self.getEstimatedParameters()

        def formatting(x):
            res = f"{x:.3g}"
            if '.' in res:
                return res
            else:
                return f"{res}.0"
        
        h += table.to_latex(float_format=formatting)

        h += "\n%%Correlation\n"
        h += "\section{Correlation}\n"
        table = self.getCorrelationResults()
        h += table.to_latex(float_format=formatting)
        return h

    def getGeneralStatistics(self):
        """ Format the results in a dict """
        d = {}
        d['Number of estimated parameters'] = self.data.nparam,''
        d['Sample size'] = self.data.sampleSize,''
        if self.data.sampleSize != self.data.numberOfObservations:
            d['Observations'] = self.data.numberOfObservations,''
        d['Excluded observations'] = self.data.excludedData,''
        d['Init log likelihood'] = self.data.initLogLike,'.7g'
        d['Final log likelihood'] = self.data.logLike,'.7g'
        d['Likelihood ratio test for the init. model'] = self.data.likelihoodRatioTest,'.7g'
        d['Rho-square for the init. model'] = self.data.rhoSquare,'.3g'
        d['Rho-square-bar for the init. model'] = self.data.rhoBarSquare,'.3g'
        d['Akaike Information Criterion'] = self.data.akaike,'.7g'
        d['Bayesian Information Criterion'] = self.data.bayesian,'.7g'
        d['Final gradient norm'] = self.data.gradientNorm,'.4E'
        d['Diagnostic'] = self.data.optimizationMessage,''
        d['Database readings'] = self.data.numberOfFunctionEval,''
        d['Iterations'] = self.data.numberOfIterations,''
        d['Data processing time'] = self.data.dataProcessingTime,''
        if self.data.monteCarlo:
            d['Number of draws'] = self.data.numberOfDraws,''
            d['Draws generation time'] = self.data.drawsProcessingTime,''
        d['Optimization time'] = self.data.optimizationTime,''
        if self.data.bootstrap is not None:
            d['Bootstrapping time'] = self.data.bootstrapTime,''
        d['Nbr of threads'] = self.data.numberOfThreads,''
        return d

    def getEstimatedParameters(self):
        columns = ['Value','Std err','t-test','p-value','Rob. Std err','Rob. t-test','Rob. p-value']
        if self.data.bootstrap is not None:
            columns += [f'Bootstrap[{len(self.data.bootstrap)}] Std err','Bootstrap t-test','Bootstrap p-value']
        table = pd.DataFrame(columns=columns)
        for b in self.data.betas:
            arow = {'Value':b.value,
                    'Std err':b.stdErr,
                    't-test':b.tTest,
                    'p-value':b.pValue,
                    'Rob. Std err':b.robust_stdErr,
                    'Rob. t-test':b.robust_tTest,
                    'Rob. p-value':b.robust_pValue}
            if self.data.bootstrap is not None:
                arow[f'Bootstrap[{len(self.data.bootstrap)}] Std err'] = b.bootstrap_stdErr
                arow['Bootstrap t-test'] = b.bootstrap_tTest
                arow['Bootstrap p-value'] = b.bootstrap_pValue
            
            table.loc[b.name] = pd.Series(arow)
        return table

    def getCorrelationResults(self):
        columns = ['Covariance','Correlation','t-test','p-value','Rob. cov.','Rob. corr.','Rob. t-test','Rob. p-value']
        if self.data.bootstrap is not None:
            columns += ['Boot. cov.','Boot. corr.','Boot. t-test','Boot. p-value']
        table = pd.DataFrame(columns=columns)
        for k,v in self.data.secondOrderTable.items():
            arow = {'Covariance':v[0],
                    'Correlation':v[1],
                    't-test':v[2],
                    'p-value':v[3],
                    'Rob. cov.':v[4],
                    'Rob. corr.':v[5],
                    'Rob. t-test':v[6],
                    'Rob. p-value':v[7]}
            if self.data.bootstrap is not None:
                arow['Boot. cov.'] = v[8]
                arow['Boot. corr.'] = v[9]
                arow['Boot. t-test'] = v[10]
                arow['Boot. p-value'] = v[11]
            table.loc[f'{k[0]}-{k[1]}'] = pd.Series(arow)
        return table
       
    def getHtml(self):
        now = datetime.datetime.now()
        h = self.getHtmlHeader()
        h += bv.getHtml()
        h += self.getHtmlFooter()
        h += "<p>This file has automatically been generated on {}</p>\n".format(now)
        h += "<p>If you drag this HTML file into the Calc application of <a href='http://www.openoffice.org/' target='_blank'>OpenOffice</a>, or the spreadsheet of <a href='https://www.libreoffice.org/' target='_blank'>LibreOffice</a>, you will be able to perform additional calculations.</p>\n"
        h += "<table>\n"
        h += "<tr class=biostyle><td align=right><strong>Report file</strong>:	</td><td>"+self.data.htmlFileName+"</td></tr>\n"
        h += "<tr class=biostyle><td align=right><strong>Database name</strong>:	</td><td>"+self.data.dataname+"</td></tr>\n"
        h += "</table>\n"

        ### Include here the part on statistics

        h += "<h1>Estimation report</h1>\n"

        h += "<table border='0'>\n"
        d = self.getGeneralStatistics()
        # k is the description of the quantity
        # v is the value
        # p is the precision to format it
        for k,(v,p) in d.items():
            h += f"<tr class=biostyle><td align=right ><strong>{k}</strong>: </td> <td>{v:{p}}</td></tr>\n"
        h += "</table>\n"

        table = self.getEstimatedParameters()

        h += "<h1>Estimated parameters</h1>\n"
        h += "<p><font size='-1'>Click on the headers of the columns to sort the table  [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</font></p>\n"
        h += "<table border='1' class='sortable'>\n"
        h += "<tr class=biostyle><th>Name</th>"
        for c in table.columns:
            h += f"<th>{c}</th>"
        h += "</tr>\n" 
        for name, values in table.iterrows():
            h += f"<tr class=biostyle><td>{name}</td>"
            for k,v in values.items():
                h += f"<td>{v:.3g}</td>"
            h += "</tr>\n"
        h += "</table>\n"

        table = self.getCorrelationResults()
        h += "<h2>Correlation of coefficients</h2>\n"
        h += "<p><font size='-1'>Click on the headers of the columns to sort the table [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</font></p>\n"
        h += "<table border='1' class='sortable'>\n"
        h += "<tr class=biostyle><th>Coefficient1</th><th>Coefficient2</th>"
        for c in table.columns:
            h += f"<th>{c}</th>"
        h += "</tr>\n" 
        for name, values in table.iterrows():
            n = name.split('-')
            h += f"<tr class=biostyle><td>{n[0]}</td><td>{n[1]}</td>"
            for k,v in values.items():
               h += f"<td>{v:.3g}</td>"
            h += "</tr>\n"
        h += "</table>\n"
                
        h += "<p>Smallest eigenvalue: {:.6g}</p>\n".format(self.data.smallestEigenValue)
        h += "<p>Smallest singular value: {:.6g}</p>\n".format(self.data.smallestSingularValue)
        return h

    def getBetaValues(self,myBetas=None):
        values = dict()
        if myBetas is None:
            myBetas = self.data.betaNames
        for b in myBetas:
            try:
                index = self.data.betaNames.index(b)
                values[b] = self.data.betas[index].value
            except:
                keys = ''
                for k in self.data.betaNames:
                    keys += f' {k}'
                err = f"The value of {b} is not available in the results. The following parameters are available: {keys}"
                raise excep.biogemeError(err)
        return values

    def getRobustVarCovar(self):
        names = [b.name for b in self.data.betas]
        vc = pd.DataFrame(index=names,columns=names)
        for i in range(len(self.data.betas)):
            for j in range(len(self.data.betas)):
                vc.at[self.data.betas[i].name, self.data.betas[j].name] = self.data.robust_varCovar[i,j]
        return vc   
    
    def writeHtml(self):
        self.data.htmlFileName = bf.getNewFileName(self.data.modelName,"html")
        f = open(self.data.htmlFileName,"w")
        f.write(self.getHtml())
        f.close()

    def writeLaTeX(self):
        self.data.latexFileName = bf.getNewFileName(self.data.modelName,"tex")
        f = open(self.data.latexFileName,"w")
        f.write(self.getLaTeX())
        f.close()
        
    def getHtmlHeader(self):
        h = ""
        h += "<html>"
        h += "<head>"
        h += "<script src='http://transp-or.epfl.ch/biogeme/sorttable.js'></script>"
        h += "<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />"
        h += "<title>"+self.data.modelName+" - Report from biogeme "+bv.getVersion()+" ["+bv.versionDate+"]</title>"
        h += "<meta name='keywords' content='biogeme, discrete choice, random utility'>"
        h += "<meta name='description' content='Report from biogeme "+bv.getVersion()+" ["+bv.versionDate+"]'>"
        h += "<meta name='author' content='"+bv.author+"'>"
        h += "<style type=text/css>"
        h += ".biostyle"
        h += "	{font-size:10.0pt;"
        h += "	font-weight:400;"
        h += "	font-style:normal;"
        h += "	font-family:Courier;}"
        h += ".boundstyle"
        h += "	{font-size:10.0pt;"
        h += "	font-weight:400;"
        h += "	font-style:normal;"
        h += "	font-family:Courier;"
        h += "        color:red}"
        h += "</style>"
        h += "</head>"
        h += "<body bgcolor='#ffffff'>"
        return h

    def getHtmlFooter(self):
        return "</html>"
        
    def getBetasForSensitivityAnalysis(self,myBetas,size=100):
        simulatedBetas = np.random.multivariate_normal(self.data.betaValues, self.data.robust_varCovar, size, check_valid='warn')
        index = [self.data.betaNames.index(b) for b in myBetas]
        return simulatedBetas[:,index]
        
    
