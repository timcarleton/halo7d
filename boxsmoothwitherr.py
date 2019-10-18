import numpy as np

def boxsmoothwitherror(y,err,size,func=np.mean):
    
    result=np.zeros(len(y))
    resulterr=np.zeros(len(y))
    for i in range(len(y)):
        lower=i-int(size/2)
        upper=i+int(np.ceil(size/2))
        if lower<0:
            lower=0
        if upper>len(y)-1:
            upper=len(y)-1
        w=np.arange(lower,upper).astype(np.int)
#        print(w)
#        print(err[w]*1.0)
        result[i]=func(y[w]/err[w]**2)/func(1/err[w]**2)
        resulterr[i]=1/np.sqrt(np.nansum(1/err[w]**2))

    return result,resulterr
