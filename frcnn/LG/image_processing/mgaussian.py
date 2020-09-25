from PIL import Image 
from pylab import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy import asarray as ar,exp
import os
import time

def imagehandle():
    start=time.clock()
    path='F:\\learning software\\python\\test3.tif'
    #path="F:\\learning software\\python\\"+filename;
    im=array(Image.open(path))
    lin=map(sum,im)
    x=array(range(len(lin)))
    #y=lin
    y=im[:,500]
    def gaussian(x,*param):
        return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+\
           param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))

    popt,pcov=curve_fit(gaussian,x,y,p0=[3,4,3,6,1,1])
    #popt,pcov=curve_fit(gaussian,x,y,p0=[80,180,600,650,100,2])
    print popt
    plt.plot(x,y)
    plt.plot(x,gaussian(x,*popt))
    end=time.clock()
    print (end-start)
    plt.legend()
    plt.show()

    imagehandle()


