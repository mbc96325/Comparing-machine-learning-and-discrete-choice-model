import numpy as np
import logging
import xml.etree.ElementTree as ET




class Expression:
    """
    This is the general expression in biogeme. It serves as a base
    class for concrete expressions.
    """
    def __init__(self):
        self.children = list()
    
    
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
        Default: returns a list with the beta parameters
        """
        s = set()
        for e in self.children:
            s = s.union(e.setOfBetas(free,fixed))
        return s

    def getLiteral(self,name):
        """
        Obtain the literal object from its name
        """
        for e in self.children:
            if e.getLiteral(name) != None:
                return e.getLiteral(name)
            
    def setRow(self,row):
        """
        Set the row of the database
        """
        self.row = row
        for e in self.children:
            e.setRow(row)

    def setBetas(self,betas):
        """
        Set the values of the parameters
        """
        self.betas = betas
        for e in self.children:
            e.setBetas(betas)

    def setIndex(self,name,index):
        for e in self.children:
            e.setIndex(name,index)

    def getXml(self):
        """
        Generates a XML version of the expression.
        """
        expression = ET.Element('bioExpression')
        expression.set('name','loglikelihood')
        self.setXmlChild(expression)
        return ET.tostring(expression)

    def setXmlChild(self,parent):
        se = ET.SubElement(parent,'bioExpression')
        se.set('name',self.getClassName())
        for e in self.children:
            sub = ET.SubElement(se,'childExpression')
            e.setXmlChild(sub)
    
    def getClassName(self):
        """
        Returns the name of the class
        """
        n = type(self).__name__
        return n


    def getSignature(self,listOfSignatures = []):
        for e in self.children:
            listOfSignatures = e.getSignature(listOfSignatures)
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '({})'.format(len(self.children))
        for e in self.children:
            signature += ',{}'.format(id(e))
        listOfSignatures.append(signature.encode())
        return listOfSignatures

    
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
        return self.left.getValue() == self.right.getValue()
    

class NotEqual(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} != {})".format(self.left,self.right)
    
    def getValue(self):
        return self.left.getValue() != self.right.getValue()
    

class LessOrEqual(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} <= {})".format(self.left,self.right)
    
    def getValue(self):
        return self.left.getValue() <= self.right.getValue()
    

class GreaterOrEqual(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} >= {})".format(self.left,self.right)
    
    def getValue(self):
        return self.left.getValue() >= self.right.getValue()
    

class Less(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} < {})".format(self.left,self.right)
    
    def getValue(self):
        return self.left.getValue() < self.right.getValue()
    

class Greater(BinaryOperator):
    def __init__(self,left,right):
        BinaryOperator.__init__(self,left,right)
        
    def __str__(self):
        return "({} > {})".format(self.left,self.right)
    
    def getValue(self):
        return self.left.getValue() > self.right.getValue()
    

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
    

    def getClassName(self):
        return "UnaryMinus"


class exp(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "exp({})".format(self.child)
        
    def getValue(self):
        return np.exp(self.child.getValue())
    

    def getClassName(self):
        return "exp"

class log(UnaryOperator):
    def __init__(self,child):
        UnaryOperator.__init__(self,child)
        
    def __str__(self):
        return "log({})".format(self.child)
        
    def getValue(self):
        return np.log(self.child.getValue())
    

    def getClassName(self):
        return "log"
    
    
class Numeric(Expression):
    """
    Simple number. 
    """
    def __init__(self,value):
        Expression.__init__(self)
        self.value = value
        
    def __str__(self):
        return str(self.value)
    
    def getValue(self):
        return self.value
    

    def getCode(self,listOfCode = []):
        code = "x{} = {}".format(id(self),self.value)
        listOfCode.append(code)
        return listOfCode
        

    def setXmlChild(self,parent):
        se = ET.SubElement(parent,'bioExpression')
        se.set('name',self.getClassName())
        se.text = "{}".format(self.value)

    def getSignature(self,listOfSignatures = []):
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += ',{}'.format(self.value)
        listOfSignatures.append(signature.encode())
        return listOfSignatures
        
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

    def setXmlChild(self,parent):
        se = ET.SubElement(parent,'bioExpression')
        se.set('name',self.getClassName())
        se.text = self.name

    def getSignature(self,listOfSignatures = []):
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '"{}"[{}]'.format(self.name,self.index)
        listOfSignatures.append(signature.encode())
        return listOfSignatures
        
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
        database.addColumn(expression,name)

    def getCode(self,listOfCode = []):
        code = "x{} = DefineVariable('{}')".format(id(self),self.name)
        listOfCode.append(code)
        return listOfCode
        
        
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

    def getCode(self,listOfCode = []):
        code = "x{} = Beta('{}',{},{},{},{},'{}')".format(id(self),self.name,self.initValue,self.lb,self.ub,self.status,self.desc)
        listOfCode.append(code)
        return listOfCode
        
    def setOfBetas(self,free=True,fixed=False):
        if fixed and self.status != 0:
            return set([self.name])
        elif free and self.status == 0:
            return set([self.name])
        else:
            return set()

    def getValue(self):
        if self.status == 0:
            return np.asscalar(self.betas[self.index])
        else:
            return self.initValue

class bioLogLogit(Expression) :
    """
    This expression is the logit formula. Although it could be built
    using other expression, its explicit implementation is designed to
    make the calculation faster, as it is used a lot in choice
    modeling.
    """
    def __init__(self, util, av, choice) :
        Expression.__init__(self)
        self.util = util
        self.av = av
        self.choice = choice
        self.children.append(self.choice)
        for i,e in self.util.items():
            self.children.append(e)
        for i,e in self.av.items():
            self.children.append(e)


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
        s = "LogLogit("
        first = True
        for i,e in self.util.items():
            if first:
                s += "{}:{}".format(int(i),e)
                first = False
            else :
                s += ",{}:{}".format(int(i),e)
        s += ")"
        return s
    
    def getSignature(self,listOfSignatures = []):
        for e in self.children:
            e.getSignature(listOfSignatures)
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '({})'.format(len(self.util))
        signature += ',{}'.format(id(self.choice))
        for i,e in self.util.items():
            signature += ',{},{},{}'.format(i,id(e),id(self.av[i]))
        listOfSignatures.append(signature.encode())
        return listOfSignatures
        

class bioMultSum(Expression):
    """
    This expression returns the sum of several other expressions. It
    is a generalization of 'Plus' for more than two terms
    """
    def __init__(self,listOfExpressions):
        Expression.__init__(self)
        for e in listOfExpressions:
            self.children.append(e)


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
        self.dictOfExpressions = dictOfExpressions
        self.keyExpression = keyExpression
        for k,v in dictOfExpressions.items():
            self.children.append(v)
        self.children.append(keyExpression)

    def getValue(self):
        key = int(self.keyExpression.getValue())
        if key in self.dictOfExpressions:
            return self.dictOfExpressions[key].getValue()
        else:
            return 0.0

    def __str__(self):
        first = True
        s = '{{'
        for k,v in self.dictOfExpressions.items():
            if first:
                s += '{}:{}'.format(k,v)
                first = False
            else:
                s += ',{}:{}'.format(k,v)
        s += '}}[{}]'.format(self.keyExpression)
        return s

    def getSignature(self,listOfSignatures = []):
        for e in self.children:
            e.getSignature(listOfSignatures)
        signature = '<{}>'.format(self.getClassName())
        signature += '{{{}}}'.format(id(self))
        signature += '({})'.format(len(self.dictOfExpressions))
        signature += ',{}'.format(id(self.keyExpression))
        for i,e in self.dictOfExpressions.items():
            signature += ',{},{}'.format(i,id(e))
        listOfSignatures.append(signature.encode())
        return listOfSignatures
    
