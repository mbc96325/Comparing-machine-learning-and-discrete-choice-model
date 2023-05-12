import numpy as np
import logging

def isNumeric(obj):
    """ Identifies if an object is numeric, that is int or float. """
    # Consider only ints and floats numeric
#    return isinstance(obj,int) or isinstance(obj,float)
    return isinstance(obj,(int,float,bool))

class Expression:
    """
    This is the general expression in biogeme. It serves as a base
    class for concrete expressions.
    """
    def __init__(self):
        self.parent = None
        self.children = list()
    
    def __add__(self,other):
        """ 
        Generate an expression for addition. Returns self + other
        """
        return Plus(self,other)
    
    def __radd__(self,other):
        """ 
        Generate an expression for addition. Returns other + self 
        """
        return Plus(other,self)
    
    
    def __sub__(self,other):
        """ 
        Generate an expression for substraction. Returns self - other
        """
        return Minus(self,other)
    
    def __rsub__(self,other):
        """ 
        Generate an expression for substraction. Returns other - self
        """
        return Minus(other,self)
    
    def __mul__(self,other):
        """ 
        Generate an expression for multiplication. Returns self * other
        """
        return Times(self,other)
    
    def __rmul__(self,other):
        """ 
        Generate an expression for multiplication. Returns other * self
        """
        return Times(other,self)
    
    # return self / other
    def __div__(self,other):
        """ 
        Generate an expression for division. Returns self / other
        """
        return Divide(self,other)
    
    def __rdiv__(self,other):
        """ 
        Generate an expression for division. Returns other / self
        """
        return Divide(other,self)
    
    def __truediv__(self,other):
        """ 
        Generate an expression for division. Returns self / other
        """
        return Divide(self,other)
    
    def __rtruediv__(self,other):
        """ 
        Generate an expression for division. Returns other / self
        """
        return Divide(other,self) 

    def __neg__(self):
        """ 
        Generate an expression for unary minus. Returns -self
        """
        return UnaryMinus(self)

    def __pow__(self,other):
        """ 
        Generate an expression for power. Returns self ^ other
        """
        return Power(self,other)

    def __rpow__(self,other):
        """ 
        Generate an expression for power. Returns other ^ self
        """
        return Power(other,self)

    def __and__(self, other):
        """ 
        Generate an expression for logical and. Returns self and other
        """
        return And(self,other)

    def __or__(self, other):
        """ 
        Generate an expression for logical or. Returns self or other
        """
        return Or(self,other)

    def __eq__(self, other):
        """ 
        Generate an expression for comparison. Returns self == other
        """
        return Equal(self,other)

    def __ne__(self, other):
        """ 
        Generate an expression for comparison. Returns self != other
        """
        return NotEqual(self,other)
    
    def __le__(self, other):
        """ 
        Generate an expression for comparison. Returns self <= other
        """
        return LessOrEqual(self,other)

    def __ge__(self, other):
        """ 
        Generate an expression for comparison. Returns self >= other
        """
        return GreaterOrEqual(self,other)
    
    def __lt__(self, other):
        """ 
        Generate an expression for comparison. Returns self < other
        """
        return Less(self,other)

    def __gt__(self, other):
        """ 
        Generate an expression for comparison. Returns self > other
        """
        return Greater(self,other)
    
    def setOfLiterals(self):
        """
        Default: returns the set of literals appearing in the expression.
        """
        s = set()
        for e in self.children:
            s = s.union(e.setOfLiterals())
        return s

    def setOfBetas(self,free=True,fixed=False):
        """
        Default: returns a set with the beta parameters
        """
        s = set()
        for e in self.children:
            s = s.union(e.setOfBetas(free,fixed))
        return s

    def dictOfBetas(self,free=True,fixed=False):
        """
        Default: returns a  dict with the beta parameters
        """
        s = {}
        for e in self.children:
            d = e.dictOfBetas(free,fixed)
            s = dict(s,**d)
        return s


    def dictOfRandomVariables(self):
        """
        Default: returns a  dict with the random variables
        """
        s = {}
        for e in self.children:
            d = e.dictOfRandomVariables()
            s = dict(s,**d)
        return s
    
    def getLiteral(self,name):
        """
        Obtain the literal object from its name
        """
        for e in self.children:
            if e.getLiteral(name) != None:
                return e.getLiteral(name)
        return None
            
    def setRow(self,row):
        """
        Set the row of the database
        """
        self.row = row
        for e in self.children:
            e.setRow(row)

    def getDraws(self):
        draws = {}
        for e in self.children:
            d = e.getDraws()
            if d:
                draws = dict(draws,**d)
        return draws
        
            
    def setBetaValues(self,dictOfValues):
        for e in self.children:
            e.setBetaValues(dictOfValues)
        return self
            
    def setIndex(self,name,index):
        for e in self.children:
            e.setIndex(name,index)

    def setDrawIndex(self,name,index):
        for e in self.children:
            e.setDrawIndex(name,index)
            
    def getClassName(self):
        """
        Returns the name of the class
        """
        n = type(self).__name__
        return n


    def getSignature(self):
        listOfSignatures = []
        for e in self.children:
            listOfSignatures += e.getSignature()
        mysignature = '<{}>'.format(self.getClassName())
        mysignature += '{{{}}}'.format(id(self))
        mysignature += '({})'.format(len(self.children))
        for e in self.children:
            mysignature += ',{}'.format(id(e))
        listOfSignatures += [mysignature.encode()]
        return listOfSignatures

    def isContainedIn(self,t):
        if self.parent is None:
            return False
        if self.parent.getClassName() == t:
            return True
        return self.parent.isContainedIn(t)
    
    def embedExpression(self,t):
        if self.getClassName() == t:
            return True
        for e in self.children:
            if e.embedExpression(t):
                return True
        return False
    
    def audit(self,database=None):
        listOfErrors = []
        listOfWarnings = []
        for e in self.children:
            err,war = e.audit(database)
            listOfErrors += err
            listOfWarnings += war
        return listOfErrors,listOfWarnings
    
class BinaryOperator(Expression):
    """
    This expression is the result of the combination of two
    expressions, typically addition, substraction,
    multiplication or division.
    """
    def __init__(self,left,right):
        Expression.__init__(self)
        if isNumeric(left):
            self.left = Numeric(left)
        else:
            self.left = left
        if isNumeric(right):
            self.right = Numeric(right)
        else:
            self.right = right
        self.left.parent = self
        self.right.parent = self
        self.children.append(self.left)
        self.children.append(self.right)
        
    def getCode(self,listOfCode = []):
        for e in self.children:
            listOfCode = e.getCode(listOfCode)
        code = "x{} = {}(x{},x{})".format(id(self),self.getClassName(),id(self.left),id(self.right))
        listOfCode.append(code)
        return listOfCode
    
       
class Plus(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} + {})".format(self.left,self.right)
    
    def getValue(self):
        return self.left.getValue() + self.right.getValue()


class Minus(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)

    def __str__(self):
        return "({} - {})".format(self.left,self.right)
    
    def getValue(self):
        return self.left.getValue() - self.right.getValue()
    
        
class Times(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} * {})".format(self.left,self.right)

    def getValue(self):
        return self.left.getValue() * self.right.getValue()
    

            
class Divide(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)

    def __str__(self):
        return "({} / {})".format(self.left,self.right)
        
    def getValue(self):
        return self.left.getValue() / self.right.getValue()
    

class Power(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)

    def __str__(self):
        return "({} ** {})".format(self.left,self.right)
        
    def getValue(self):
        return self.left.getValue() ** self.right.getValue()

class bioMin(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "bioMin({},{})".format(self.left,self.right)
    
    def getValue(self):
        if (self.left.getValue() <= self.right.getValue()):
            return self.left.getValue()
        else:
            return self.right.getValue()

class bioMax(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "bioMax({},{})".format(self.left,self.right)
    
    def getValue(self):
        if (self.left.getValue() >= self.right.getValue()):
            return self.left.getValue()
        else:
            return self.right.getValue()
        
        
class And(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} and {})".format(self.left,self.right)
    
    def getValue(self):
        if (self.left.getValue() == 0.0):
            return 0.0
        if (self.right.getValue() == 0.0):
            return 0.0
        return 1.0
    

class Or(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} or {})".format(self.left,self.right)
    
    def getValue(self):
        if (self.left.getValue() != 0.0):
            return 1.0
        if (self.right.getValue() != 0.0):
            return 1.0
        return 0.0
    

class Equal(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} == {})".format(self.left,self.right)
    
    def getValue(self):
        r = 1 if self.left.getValue() == self.right.getValue() else 0
        return r
    

class NotEqual(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} != {})".format(self.left,self.right)
    
    def getValue(self):
        r = 1 if self.left.getValue() != self.right.getValue() else 0
        return r
    

class LessOrEqual(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} <= {})".format(self.left,self.right)
    
    def getValue(self):
        r = 1 if self.left.getValue() <= self.right.getValue() else 0
        return r
    

class GreaterOrEqual(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} >= {})".format(self.left,self.right)
    
    def getValue(self):
        r = 1 if self.left.getValue() >= self.right.getValue() else 0
        return r
    

class Less(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} < {})".format(self.left,self.right)
    
    def getValue(self):
        r = 1 if self.left.getValue() < self.right.getValue() else 0
        return r
    

class Greater(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} > {})".format(self.left,self.right)
    
    def getValue(self):
        r = 1 if self.left.getValue() > self.right.getValue() else 0
        return r
    

class UnaryOperator(Expression):
    """
    This expression is the result of the modification of another
    expressions, typically changing its sign.
    """
    def __init__(self,child):
        Expression.__init__(self)
        if isNumeric(child):
            self.child = Numeric(child)
        else :
            self.child = child
        self.child.parent = self
        self.children.append(self.child)
            

    def getCode(self,listOfCode = []):
        for e in self.children:
            listOfCode = e.getCode(listOfCode)
        code = "x{} = {}(x{})".format(id(self),self.getClassName(),id(child))
        listOfCode.append(code)
        return listOfCode
        
    
class UnaryMinus(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "(-{})".format(self.child)
        
    def getValue(self):
        return - self.child.getValue() 

class MonteCarlo(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "MonteCarlo({})".format(self.child)
        
    def getValue(self):
        raise excep.biogemeError("Not implemented in Python. Use the C++ implementation.")
    def audit(self,database=None):
        listOfErrors,listOfWarnings = self.child.audit(database)
        if not self.child.embedExpression("bioDraws"):
            theError = "The argument of MonteCarlo must contain a bioDraws: {}".format(self)
            listOfErrors.append(theError)
        if self.child.embedExpression("MonteCarlo"):
            theError = "It is not possible to include a MonteCarlo statement in another one: {}".format(self)
            listOfErrors.append(theError)
        return listOfErrors,listOfWarnings
        

class bioNormalCdf(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "bioNormalCdf({})".format(self.child)
        
    def getValue(self):
        raise excep.biogemeError("Not implemented in Python. Use the C++ implementation.")
        

class PanelLikelihoodTrajectory(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "PanelLikelihoodTrajectory({})".format(self.child)
        
    def getValue(self):
        raise biogemeError("Not implemented in Python. Use the C++ implementation.")
    def audit(self,database=None):
        listOfErrors = []
        listOfWarnings = []
        if not database.isPanel():
            theError = "Expression PanelLikelihoodTrajectory can only be used with panel data. Use the statement database.panel('IndividualId') to declare the panel strucutre of the data: {}".format(self)
            listOfErrors.append(theError)
        return listOfErrors,listOfWarnings
    


class exp(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "exp({})".format(self.child)
        
    def getValue(self):
        return np.exp(self.child.getValue())
    

class log(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "log({})".format(self.child)
        
    def getValue(self):
        return np.log(self.child.getValue())

    
class Derive(UnaryOperator):
    def __init__(self,child,name):
        UnaryOperator.__init__(self,child)
        self.literalName = name
        self.literalIndex = -1  

    def setIndex(self,name,index):
        if self.literalName == name:
            self.literalIndex = index
        self.child.setIndex(name,index)

    def getSignature(self):
        listOfSignatures = []
        listOfSignatures += self.child.getSignature()
        mysignature = '<{}>'.format(self.getClassName())
        mysignature += '{{{}}}'.format(id(self))
        mysignature += ',{}'.format(id(self.child))
        mysignature += ',{}'.format(self.literalIndex)
        listOfSignatures += [mysignature.encode()]
        return listOfSignatures
            
    def __str__(self):
        return "Derive({},'{}')".format(self.child,self.literalName)
        
    def getValue(self):
        raise biogemeError("Not implemented in Python. Use the C++ implementation.")
    

class Integrate(UnaryOperator):
    def __init__(self,child,name):
        UnaryOperator.__init__(self,child)
        self.literalName = name
        self.literalIndex = -1  

    def setIndex(self,name,index):
        if self.literalName == name:
            self.literalIndex = index
        self.child.setIndex(name,index)

    def getSignature(self):
        listOfSignatures = []
        listOfSignatures += self.child.getSignature()
        mysignature = '<{}>'.format(self.getClassName())
        mysignature += '{{{}}}'.format(id(self))
        mysignature += ',{}'.format(id(self.child))
        mysignature += ',{}'.format(self.literalIndex)
        listOfSignatures += [mysignature.encode()]
        return listOfSignatures
            
    def __str__(self):
        return "Integrate({},'{}')".format(self.child,self.literalName)
        
    def getValue(self):
        raise biogemeError("Not implemented in Python. Use the C++ implementation.")
    

class bioDraws(Expression):
    def __init__(self,name,type):
        Expression.__init__(self)
        self.name = name
        self.type = type
        self.drawIndex = -1

    def __str__(self):
        return "bioDraws('{}','{}')".format(self.name,self.type)
    
    def getValue(self):
        raise biogemeError("Not implemented in Python. Use the C++ implementation.")

    def getCode(self,listOfCode = []):
        code = "x{} = {}".format(id(self),self.value)
        listOfCode.append(code)
        return listOfCode
        
    def getSignature(self):
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '"{}"[{}]'.format(self.name,self.drawIndex)
        return [signature.encode()]

    def getDraws(self):
        return {self.name: self.type}

    def setDrawIndex(self,name,index):
        if self.name == name:
            self.drawIndex = index

    def audit(self,database=None):
        listOfErrors = []
        listOfWarnings = []
        if not self.isContainedIn("MonteCarlo"):
            theError = "bioDraws expression must be embedded into a MonteCarlo: {}".format(self)
            listOfErrors.append(theError)
        return listOfErrors,listOfWarnings
            
    
class Numeric(Expression):
    """
    Simple number. 
    """
    def __init__(self,value):
        Expression.__init__(self)
        self.value = value
        
    def __str__(self):
        return "`"+str(self.value)+"`"
    
    def getValue(self):
        return self.value
    

    def getCode(self,listOfCode = []):
        code = "x{} = {}".format(id(self),self.value)
        listOfCode.append(code)
        return listOfCode
        
    def getSignature(self):
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += ',{}'.format(self.value)
        return [signature.encode()]
        
class Literal(Expression):
    """
    Name appearing in an expression. In bioeme, it can be a variable
    (from the satabase), or a parameter (that must be estimated using
    maximum likelihood.
    """
    def __init__(self,name):
        Expression.__init__(self)
        self.name = name
        self.index = -1

    def __str__(self):
        return self.name

    def setOfLiterals(self):
        return set([self.name])


    def getLiteral(self,name):
        """
        Obtain the literal object from its name
        """
        if self.name == name:
            return self
        else:
            return None
    

    def setIndex(self,name,index):
        if self.name == name:
            self.index = index

    def getSignature(self):
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '"{}"[{}]'.format(self.name,self.index)
        return [signature.encode()]
        
class Variable(Literal):
    """
    This represents the explanatory variables of the choice
    model. Typically, they come from the data set.
    """
    def __init__(self,name):
        Literal.__init__(self,name)
    
    def getValue(self):
        return self.row[self.name]

    def getCode(self,listOfCode = []):
        code = "x{} = Variable('{}')".format(id(self),self.name)
        listOfCode.append(code)
        return listOfCode


class DefineVariable(Variable):
    """
    This expression allows the use to define a new variable that will
    be added to the database. It avoids that it is recalculated each
    time it is needed.
    """
    def __init__(self,name,expression,database):
        Variable.__init__(self,name)
        if isNumeric(expression):
            database.addColumn(Numeric(expression),name)
        else:
            database.addColumn(expression,name)

    def getCode(self,listOfCode = []):
        code = "x{} = DefineVariable('{}')".format(id(self),self.name)
        listOfCode.append(code)
        return listOfCode


class RandomVariable(Variable):
    """
    This expression allows the use to define a random variable used for integration
    """
    def __init__(self,name):
        Variable.__init__(self,name)

    def getCode(self,listOfCode = []):
        code = "x{} = RandomVariable('{}')".format(id(self),self.name)
        listOfCode.append(code)
        return listOfCode

    def dictOfRandomVariables(self):
        return {self.name: self}
        
class Beta(Literal):
    """
    This represents the parameters of the model, that have to be
    estimated using maximum likelihood
    """
    def __init__(self,name,value,lowerbound,upperbound,status,desc=''):
        Literal.__init__(self,name)
        self.initValue = value
        self.lb = lowerbound
        self.ub = upperbound
        self.status = status
        self.desc = desc

    def __str__(self):
        return "{}({})".format(self.name,self.initValue)
        
    def getCode(self,listOfCode = []):
        code = "x{} = Beta('{}',{},{},{},{},'{}')".format(id(self),self.name,self.initValue,self.lb,self.ub,self.status,self.desc)
        listOfCode.append(code)
        return listOfCode
        
    def setBetaValues(self,dictOfValues):
        v = dictOfValues.get(self.name)
        if v is not None:
            self.initValue = v
            
    def setOfBetas(self,free=True,fixed=False):
        if fixed and self.status != 0:
            return set([self.name])
        elif free and self.status == 0:
            return set([self.name])
        else:
            return set()

    def dictOfBetas(self,free=True,fixed=False):
        if fixed and self.status != 0:
            return {self.name:self}
        elif free and self.status == 0:
            return {self.name:self}
        else:
            return dict()
        
    def getValue(self):
        return self.initValue

class LogitLike(Expression) :
    """
    This expression captures the logit formula. It contains one formula for the target alternative, a dict of formula for the availabilities and a dict of formulas for the utilities
    """
    def __init__(self, util, av, choice) :
        Expression.__init__(self)
        self.util = {}
        for i,e in util.items():
            if isNumeric(e):
                self.util[i] = Numeric(e)
            else:
                self.util[i] = e
        self.av = {}
        for i,e in av.items():
            if isNumeric(e):
                self.av[i] = Numeric(e)
            else:
                self.av[i] = e
        if isNumeric(choice):
            self.choice = Numeric(choice)
        else:
            self.choice = choice

        self.choice.parent = self
        self.children.append(self.choice)
        for i,e in self.util.items():
            e.parent = self
            self.children.append(e)
        for i,e in self.av.items():
            e.parent = self
            self.children.append(e)

    def audit(self,database):
        listOfErrors = []
        listOfWarnings = []
        for e in self.children:
            err,war = e.audit(database)
            listOfErrors += err
            listOfWarnings += war

        listOfAlternatives = list(self.util)
        choices = database.valuesFromDatabase(self.choice)
        correctChoices = choices.isin(listOfAlternatives)
        indexOfIncorrectChoices = correctChoices.index[correctChoices == False].tolist()
        if indexOfIncorrectChoices:
            incorrectChoices = choices[indexOfIncorrectChoices]
            theError = "The choice variable [{}] does not correspond to a valid alternative for the following observations (rownumber[choice]): ".format(self.choice)+'-'.join('{}[{}]'.format(*t) for t in zip(indexOfIncorrectChoices,incorrectChoices))
            listOfErrors.append(theError)

        choiceAvailability = database.checkAvailabilityOfChosenAlt(self.av,self.choice)
        indexOfUnavailableChoices = choiceAvailability.index[choiceAvailability == False].tolist()
        if indexOfUnavailableChoices:
            incorrectChoices = choices[indexOfUnavailableChoices]
            theError = "The chosen alternative is not available for the following observations (rownumber[choice]): ".format(self.choice)+'-'.join('{}[{}]'.format(*t) for t in zip(indexOfUnavailableChoices,incorrectChoices))
            listOfWarnings.append(theError)
        
        return listOfErrors,listOfWarnings


    def getCode(self,listOfCode = []):
        for e in self.children:
            listOfCode = e.getCode(listOfCode)
        codeV = "V{} = {{"+",".join('{} : {}'.format(key, id(value)) for key, value in self.util.items())+"}}"
        codeA = "av{} = {{"+",".join('{} : {}'.format(key, id(value)) for key, value in self.util.items())+"}}"
        code = "x{} = bioLogLogit(V{},av{},x{})".format(id(self),id(self),id(self),id(self.choice))
        listOfCode.append(code)
        return listOfCode
            
    def getValue(self):
        choice = int(self.choice.getValue())
        if choice not in self.util:
            logging.warning("Choice is {}. List of alternatives is {}".format(choice,self.util.keys()))
            return np.nan
        if self.av[choice].getValue() == 0.0:
            return -np.log(0)
        Vchosen = self.util[choice].getValue()
        denom = 0.0
        for i,V in self.util.items():
            if self.av[i].getValue() != 0.0:
                denom += np.exp(V.getValue()-Vchosen)
        return -np.log(denom)
        



    def __str__(self):
        s = self.getClassName()
        first = True
        for i,e in self.util.items():
            if first:
                s += "{}:{}".format(int(i),e)
                first = False
            else :
                s += ",{}:{}".format(int(i),e)
        s += ")"
        return s
    
    def getSignature(self):
        listOfSignatures = []
        for e in self.children:
            listOfSignatures += e.getSignature()
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '({})'.format(len(self.util))
        signature += ',{}'.format(id(self.choice))
        for i,e in self.util.items():
            signature += ',{},{},{}'.format(i,id(e),id(self.av[i]))
        listOfSignatures += [signature.encode()]
        return listOfSignatures
        

class bioLogLogit(LogitLike):
    pass

class bioMultSum(Expression):
    """
    This expression returns the sum of several other expressions. It
    is a generalization of 'Plus' for more than two terms
    """
    def __init__(self,listOfExpressions):
        Expression.__init__(self)
        if type(listOfExpressions) is dict:
            for k,e in listOfExpressions.items():
                if isNumeric(e):
                    theExpression = Numeric(e)
                    theExpression.parent = self
                    self.children.append(theExpression)
                else:
                    e.parent = self
                    self.children.append(e)
        elif type(listOfExpressions) is list:
            for e in listOfExpressions:
                if isNumeric(e):
                    theExpression = Numeric(e)
                    theExpression.parent = self
                    self.children.append(theExpression)
                else:
                    e.parent = self
                    self.children.append(e)
        else:
            raise excep.biogemeError("Argument of bioMultSum must be a dict or a list.")


    def getCode(self,listOfCode = []):
        for e in self.children:
            listOfCode = e.getCode(listOfCode)
        code = "x{} = {}(".format(id(self),self.getClassName())
        code += ",".join('{}'.format(id(f)) for f in self.children)
        code += ")"
        listOfCode.append(code)
        return listOfCode
        
    def getValue(self):
        result = 0.0
        for e in self.children:
            result += e.getValue()
        return result

    def __str__(self):
        first = True
        s = ''
        for e in self.children:
            if first:
                s += "({}".format(e)
                first = False
            else:
                s += " + {}".format(e)
        s += ')'
        return s
    
class Elem(Expression):
    """
    This returns the element of a dictionary. The key is evaluated
    from an expression.
    """

    def __init__(self,dictOfExpressions,keyExpression):
        Expression.__init__(self)
        self.dictOfExpressions = {}
        for k,v in dictOfExpressions.items():
            if isNumeric(v):
                self.dictOfExpressions[k] = Numeric(v)
            else:
                self.dictOfExpressions[k] = v
            self.dictOfExpressions[k].parent = self                
            self.children.append(self.dictOfExpressions[k])
                
        if type(keyExpression) is bool:
            self.keyExpression = Numeric(1) if keyExpression else Numeric(0)
        elif isNumeric(keyExpression):
            self.keyExpression = Numeric(keyExpression)
        else:
            self.keyExpression = keyExpression
        self.keyExpression.parent = self
        self.children.append(self.keyExpression)

    def getValue(self):
        key = int(self.keyExpression.getValue())
        if key in self.dictOfExpressions:
            return self.dictOfExpressions[key].getValue()
        else:
            return 0.0

    def __str__(self):
        s = '{{'
        first = True
        for k,v in self.dictOfExpressions.items():
            if first:
                s += '{}:{}'.format(k,v)
                first = False
            else:
                s += ',{}:{}'.format(k,v)
        s += '}}[{}]'.format(self.keyExpression)
        return s

    def getSignature(self):
        listOfSignatures = []
        listOfSignatures += self.keyExpression.getSignature()
        for i,e in self.dictOfExpressions.items():
            listOfSignatures += e.getSignature()
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '({})'.format(len(self.dictOfExpressions))
        signature += ',{}'.format(id(self.keyExpression))
        for i,e in self.dictOfExpressions.items():
            signature += ',{},{}'.format(i,id(e))
        listOfSignatures += [signature.encode()]
        return listOfSignatures
    
