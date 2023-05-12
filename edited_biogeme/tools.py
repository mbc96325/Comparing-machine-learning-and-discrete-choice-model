import numpy as np
import logging

# Calculate the gradient with finite differences
def findiff_g(theFunction,x):
    tau = 0.0000001
    n = len(x)
    g = np.zeros(n)
    f = theFunction(x)[0]
    for i in range(n):
        xi = x.item(i)
        xp = x.copy()
        if (abs(xi) >= 1):
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        xp[i] = xi + s
        fp = theFunction(xp)[0]

        g[i] = (fp - f) / s
    
    return g

# Calculate the Hessian with finite differences
def findiff_H(theFunction,x):
    tau = 0.0000001
    n = len(x)
    H = np.zeros((n,n))
    g = theFunction(x)[1]
    I = np.eye(n,n)
    for i in range(n):
        xi = x.item(i)
        if (abs(xi) >= 1):
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        ei = I[i]
        gp = theFunction(x + s * ei)[1]
        H[:,i] = (gp-g).flatten() / s

    return H


def checkDerivatives(theFunction,x,names,logg=False):
    f,g,h = theFunction(x)
    g_num = findiff_g(theFunction,x)
    gdiff = g - g_num
    if logg:
        logging.info("x\t\tGradient\tFinDiff\t\tDifference")
        for k in range(len(gdiff)):
            logging.info("{:15}\t{:+E}\t{:+E}\t{:+E}".format(names[k],g.item(k),g_num.item(k),gdiff.item(k)))

    h_num = findiff_H(theFunction,x)
    hdiff = h - h_num
    if logg:
        logging.info("Row\t\tCol\t\tHessian\tFinDiff\t\tDifference")
        for row in range(len(hdiff)):
            for col in range(len(hdiff)):
                logging.info("{:15}\t{:15}\t{:+E}\t{:+E}\t{:+E}".format(names[row],names[col],h[row,col],h_num[row,col],hdiff[row,col]))
    return f,g,h,gdiff,hdiff
                    
