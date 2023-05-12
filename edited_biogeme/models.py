from expressions import *

def logit(V,av,i):
    return exp(bioLogLogit(V,av,i))

def boxcox(x,l):
    return (x**l-1.0)/l

def piecewise(variable,thresholds):
    """ Generate the variables to include in a piecewise linear specification """
    n = len(thresholds)
    results = [bioMin(variable,thresholds[0])]
    for i in range(0,n-1):
        b = thresholds[i+1]-thresholds[i]
        results += [bioMax(Numeric(0),bioMin(variable - thresholds[i],b))]
    results += [bioMax(0,variable - thresholds[-1])]
    return results
    
## Choice probability for a MEV model.
# @ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function. If an alternative \f$i\f$ is not available, then \f$G_i = 0\f$.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param choice Expression producing the id of the chosen alternative.
# @return Choice probability of the MEV model, given by
#  \f[
#    \frac{e^{V_i + \ln G_i(e^{V_1},\ldots,e^{V_J})}}{\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}}
#  \f]
#
# \code
# def mev(V,Gi,av,choice) :
#     H = {}
#     for i,v in V.items() :
#        H[i] =  Elem({0:0, 1: v + log(Gi[i])},Gi[i]!=0)  
#     P = logit(H,av,choice)
#     return P
# \endcode
def mev(V,Gi,av,choice) :
    H = {}
    for i,v in V.items() :
        H[i] =  Elem({0:0, 1: v + log(Gi[i])},av[i]!=0)  
    P = logit(H,av,choice)
    return P

## Log of the choice probability for a MEV model.
# @ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function. If an alternative \f$i\f$ is not available, then \f$G_i = 0\f$.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param choice Expression producing the id of the chosen alternative.
# @return Log of the choice probability of the MEV model, given by
#  \f[
#    V_i + \ln G_i(e^{V_1},\ldots,e^{V_J}) - \log\left(\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}\right)
#  \f]
#
# \code
# def logmev(V,Gi,av,choice) :
#     H = {}
#     for i,v in V.items() :
#        H[i] =  Elem({0:0, 1: v + log(Gi[i])},Gi[i]!=0)  
#     P = bioLogLogit(H,av,choice)
#     return P
# \endcode
def logmev(V,Gi,av,choice) :
    H = {}
    for i,v in V.items() :
        H[i] =  Elem({0:0, 1: v + log(Gi[i])},av[i]!=0)  
    logP = bioLogLogit(H,av,choice)
    return logP



## Choice probability for a MEV model, including the correction for endogenous sampling as proposed by <a href="http://dx.doi.org/10.1016/j.trb.2007.09.003" taret="_blank">Bierlaire, Bolduc and McFadden (2008)</a>.
# @ingroup biogeme
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param correction A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the correction. Typically, it is a value, or a
# parameter to be estimated.
# @param choice Expression producing the id of the chosen alternative.
# @return Choice probability of the MEV model, given by
#  \f[
#    \frac{e^{V_i + \ln G_i(e^{V_1},\ldots,e^{V_J})}}{\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}}
#  \f]
#
# \code
# def mev_selectionBias(V,Gi,av,correction,choice) :
#     H = {}
#     for i,v in V.items() :
#         H[i] = v + log(Gi[i]) + correction[i]
#     P = logit(H,av,choice)
#     return P
# \endcode
def mev_selectionBias(V,Gi,av,correction,choice) :
    H = {}
    for i,v in V.items() :
        H[i] = v + log(Gi[i]) + correction[i]

    P = logit(H,av,choice)
            
    return P



## Log of choice probability for a MEV model, including the correction for endogenous sampling as proposed by <a href="http://dx.doi.org/10.1016/j.trb.2007.09.003" taret="_blank">Bierlaire, Bolduc and McFadden (2008)</a>.
# @ingroup biogeme
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param correction A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the correction. Typically, it is a value, or a
# parameter to be estimated.
# @param choice Expression producing the id of the chosen alternative.
# @return Log of choice probability of the MEV model, given by
#  \f[
#    V_i + \ln G_i(e^{V_1},\ldots,e^{V_J}) - \log\left(\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}\right)
#  \f]
#
# \code
# def logmev_selectionBias(V,Gi,av,correction,choice) :
#     H = {}
#     for i,v in V.items() :
#         H[i] = v + log(Gi[i]) + correction[i]
#     P = bioLogLogit(H,av,choice)
#     return P
# \endcode
def logmev_selectionBias(V,Gi,av,correction,choice) :
    H = {}
    for i,v in V.items() :
        H[i] = v + log(Gi[i]) + correction[i]

    P = bioLogLogit(H,av,choice)
            
    return P

## Implements the MEV generating function for the nested logit model
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @return A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J}) = e^{(\mu_m-1)V_i} \left(\sum_{i=1}^{J_m} e^{\mu_m V_i}\right)^{\frac{1}{\mu_m}-1}
#\f]
# where \f$m\f$ is the (only) nest containing alternative \f$i\f$, and
# \f$G\f$ is the MEV generating function.
#
def getMevForNested(V,availability,nests) :

    y = {}
    for i,v in V.items() :
        y[i] = exp(v)
    
    Gi = {}
    for m in nests:
        sumdict = list()
        for i in m[1]:
            sumdict.append(Elem({0:0.0,1: y[i] ** m[0]},availability[i]!=0))
        sum = bioMultSum(sumdict)
        for i in m[1]:
            Gi[i] = Elem({0:0,1:y[i]**(m[0]-1.0) * sum ** (1.0/m[0] - 1.0)},availability[i]!=0)
    return Gi


## Implements the nested logit model as a MEV model. 
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @return Choice probability for the nested logit model, based on the
# derivatives of the MEV generating function produced by the function
# nested::getMevForNested
#
def nested(V,availability,nests,choice) :
    Gi = getMevForNested(V,availability,nests)
    P = mev(V,Gi,availability,choice) 
    return P

## Implements the log of a nested logit model as a MEV model. 
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @return Log of choice probability for the nested logit model, based on the
# derivatives of the MEV generating function produced by the function
# nested::getMevForNested
#
def lognested(V,availability,nests,choice) :
    Gi = getMevForNested(V,availability,nests)
    logP = logmev(V,Gi,availability,choice) 
    return logP



## Implements the nested logit model as a MEV model, where mu is also a
## parameter, if the user wants to test different normalization
## schemes.
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @param mu expression producing the value of the top-level scale parameter.
# @return The nested logit choice probability based on the following derivatives of the MEV generating function: 
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J}) = \mu e^{(\mu_m-1)V_i} \left(\sum_{i=1}^{J_m} e^{\mu_m V_i}\right)^{\frac{\mu}{\mu_m}-1}
#\f]
# where \f$m\f$ is the (only) nest containing alternative \f$i\f$, and
# \f$G\f$ is the MEV generating function.
#
def nestedMevMu(V,availability,nests,choice,mu) :

    y = {}
    for i,v in V.items() :
        y[i] = exp(v)
    
    Gi = {}
    for m in nests:
        sum = list()
        for i in m[1]:
            sum.append(Elem({0:0,1: y[i] ** m[0]},availability[i]!=0))
        for i in m[1]:
            Gi[i] = Elem({0:0,1:mu * y[i]**(m[0]-1.0) * bioMultSum(sum) ** (mu/m[0] - 1.0)},availability[i]!=0)
    P = mev(V,Gi,availability,choice) 
    return P


## Implements the log of the nested logit model as a MEV model, where mu is also a
## parameter, if the user wants to test different normalization
## schemes.
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @param mu expression producing the value of the top-level scale parameter.
# @return The nested logit choice probability based on the following derivatives of the MEV generating function: 
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J}) = \mu e^{(\mu_m-1)V_i} \left(\sum_{i=1}^{J_m} e^{\mu_m V_i}\right)^{\frac{\mu}{\mu_m}-1}
#\f]
# where \f$m\f$ is the (only) nest containing alternative \f$i\f$, and
# \f$G\f$ is the MEV generating function.
#
def lognestedMevMu(V,availability,nests,choice,mu) :

    y = {}
    for i,v in V.items() :
        y[i] = exp(v)
    
    Gi = {}
    for m in nests:
        sum = list()
        for i in m[1]:
            sum.append(Elem({0:0,1: y[i] ** m[0]},availability[i]!=0))
        for i in m[1]:
            Gi[i] = Elem({0:0,1:mu * y[i]**(m[0]-1.0) * bioMultSum(sum) ** (mu/m[0] - 1.0)},availability[i]!=0)
    logP = logmev(V,Gi,availability,choice) 
    return logP

## Implements the cross-nested logit model as a MEV model. 
# \ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest. 
# Example with two nests and 6 alternatives:
# \code
#alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
#alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
#nesta = MUA , alphaA
#nestb = MUB , alphaB
#nests = nesta, nestb
# \endcode
# \return Choice probability for the cross-nested logit model.
#
def cnl_avail(V,availability,nests,choice) :
    Gi = {}
    Gidict = {}
    for k in V:
        Gidict[k] = list()
    for m in nests:
        biosumlist = list()
        for i,a in m[1].items():
            biosumlist.append(Elem({0:0,1:a**(m[0]) * exp(m[0] * (V[i]))},availability[i] != 0))
        biosum = bioMultSum(biosumlist)        
        for i,a in m[1].items():
            Gidict[i].append(Elem({0:0,1:(biosum**((1.0/m[0])-1.0)) * (a**m[0]) * exp((m[0]-1.0)*(V[i]))},availability[i] != 0))
    for k in V:
        Gi[k] = bioMultSum(Gidict[k])
    P = mev(V,Gi,availability,choice) 
    return P


## Implements the log of the cross-nested logit model as a MEV model. 
# \ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest. 
# Example with two nests and 6 alternatives:
# \code
#alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
#alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
#nesta = MUA , alphaA
#nestb = MUB , alphaB
#nests = nesta, nestb
# \endcode
# \return Choice probability for the cross-nested logit model.
#
def logcnl_avail(V,availability,nests,choice) :
    Gi = {}
    Gidict = {}
    for k in V:
        Gidict[k] = list()
    for m in nests:
        biosumlist = list()
        for i,a in m[1].items():
            biosumlist.append(Elem({0:0,1:a**(m[0]) * exp(m[0] * (V[i]))},availability[i] != 0))
        biosum = bioMultSum(biosumlist)
        for i,a in m[1].items():
            Gidict[i].append(Elem({0:0,1:(biosum**((1.0/m[0])-1.0)) * (a**m[0]) * exp((m[0]-1.0)*(V[i]))},availability[i] != 0))
    for k in V:
        Gi[k] = bioMultSum(Gidict[k])
    logP = logmev(V,Gi,availability,choice) 
    return logP


## Implements the cross-nested logit model as a MEV model with the homogeneity parameters is explicitly involved
# \ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest. 
# Example with two nests and 6 alternatives:
# \code
#alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
#alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
#nesta = MUA , alphaA
#nestb = MUB , alphaB
#nests = nesta, nestb
# \endcode
# \param bmu Homogeneity parameter \f$\mu\f$.
# \return Choice probability for the cross-nested logit model.
def cnlmu(V,availability,nests,choice,bmu) :
    Gi = {}
    Gidict = {}
    for k in V:
        Gilist[k] = list()
    for m in nests:
        biosumlist = list()
        for i,a in m[1].items():
            biosumlist.append(Elem({0:0,1:a**(m[0]/bmu) * exp(m[0] * (V[i]))},availability[i] != 0))
        biosum = bioMultSum(biosumlist)
        for i,a in m[1].items():
            Gilist[i].append(Elem({0:0,1:bmu * (biosum**((bmu/m[0])-1.0)) * (a**(m[0]/bmu)) * exp((m[0]-1.0)*(V[i]))},availability[i] != 0))
    for k in V:
        Gi[k] = bioMultSum(Gilist[k])
    P = mev(V,Gi,availability,choice) 
    return P

## Implements the log of the cross-nested logit model as a MEV model with the homogeneity parameters is explicitly involved
# \ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest. 
# Example with two nests and 6 alternatives:
# \code
#alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
#alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
#nesta = MUA , alphaA
#nestb = MUB , alphaB
#nests = nesta, nestb
# \endcode
# \param bmu Homogeneity parameter \f$\mu\f$.
# \return Log of choice probability for the cross-nested logit model.
def logcnlmu(V,availability,nests,choice,bmu) :
    Gi = {}
    Gidict = {}
    for k in V:
        Gidict[k] = list()
    for m in nests:
        biosumlist = list() 
        for i,a in m[1].items():
            biosumlist.append(Elem({0:0,1:a**(m[0]/bmu) * exp(m[0] * (V[i]))},availability[i] != 0))
        biosum = bioMultSum(biosumlist)
        for i,a in m[1].items():
            Gidict[i].append(Elem({0:0,1:bmu * (biosum**((bmu/m[0])-1.0)) * (a**(m[0]/bmu)) * exp((m[0]-1.0)*(V[i]))},availability[i] != 0))
    for k in V:
        Gi[k] = bioMultSum(Gidict[k])
    logP = logmev(V,Gi,availability,choice) 
    return logP

