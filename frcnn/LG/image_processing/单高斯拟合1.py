from PIL import Image 
from pylab import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy import asarray as ar,exp

def imagehandle(filename):
    #path='F:\\learning software\\python\\test3.tif'
    path="F:\\learning software\\python\\"+filename;
    im=array(Image.open(path))
    lin=map(sum,im)
    x=array(range(len(lin)))
    y=lin
    def gaussian(x,*param):
        return param[0]*np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))
    popt,pcov=curve_fit(gaussian,x,y,p0=[3,3,1])
    print popt
    plt.plot(x,y)
    plt.plot(x,gaussian(x,*popt))
    plt.legend()
    plt.show()
