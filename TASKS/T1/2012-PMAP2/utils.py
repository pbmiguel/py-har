### statistical fe
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import ndimage
from scipy.stats import iqr
import numpy as np
from astropy.stats import median_absolute_deviation
from numpy.random import randn
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import pearsonr

def ske(args):
    # skeweness
    if len(args) != 1:
        raise BaseException("Can Only Handle With One Array")
    return skew(args[0]);
def kur(args):
    # kurtosis
    if len(args) != 1:
        raise BaseException("Can Only Handle With One Array")
    return kurtosis(args[0])
#*
def his(args):
    # histogram
    if len(args) != 1:
        raise BaseException("Can Only Handle With One Array")
    return np.histogram(args[0])
# 

### time-series fe
def avg(args):
    # mean
    out = []
    for arg in args:
        out += arg
    return np.mean(out);
def maxi(args):
    #max
    out = []
    for arg in args:
        out += arg
    return np.array(out).max();
def mini(args):
    #minimum
    out = []
    for arg in args:
        out += arg
    return np.array(out).min();
def var(args):
    #variance https://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.var.html
    out = []
    for arg in args:
        out += arg
    return np.var(out);
#*
def cen(args):
    #centroid
    if len(args) == 1:
        return float(ndimage.measurements.center_of_mass(pd.Series(args[0]))[0]);
    else:
        raise BaseException("Invalid Arguments"); 
        
def std(args):
    # standard deviation https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.std.html
    out = []
    for arg in args:
        out += arg
    return np.std(out);
def rms(args):
    # root mean square
    out = []
    for arg in args:
        out += arg
    return np.sqrt(np.mean(pd.Series(out)**2))
# * 
def iqrr(args):
    #interquartile range
    out = []
    for arg in args:
        out += arg
    return iqr(pd.Series(args[0]))
def mad(args):
    #mean absolute deviation
    out = []
    for arg in args:
        out += arg
    return median_absolute_deviation(out)
#*
def zcr(args):
    raise BaiseException("Not Yet Implemented")
    #zero crossing rate
    out = args[0]
    for arg in args:
        out += arg
    return ((out[:-1] * out[1:]) < 0).sum()
#*
def cec(args):
    raise BaiseException("Not Yet Implemented")
    #cepstral coefficients https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    out = args[0]
    for arg in args:
        out += arg
    y, sr = librosa.load(librosa.util.example_audio_file())
    return librosa.feature.mfcc(y=y, sr=sr)
def cor(args):
    #pearson correlation
    #if len(args) == 2:
    #    return args[0].corr(args[1], method='pearson')
    #else:
    #    raise BaseException("Invalid Arguments");        
    # x-y
    if len(args) == 2:
        return float(pearsonr(args[0], args[1])[0]);
    else:
        raise BaseException("Invalid Arguments");        

def aco(args):
    #autocorrelation
    if len(args) == 1:
        if not isinstance(args[0], pd.Series):
            args[0] = pd.Series(args[0])
        #print(type(args[0]))
        return args[0].autocorr()
    else:
        raise BaseException("Invalid Arguments");    

'''
data = pd.Series([1,2,3,4,5,6]);
print(ske([[1,2]]))
print(kur([[1,2]]))
print(avg([[1,2]]))
print(maxi([[1,2]]))
print(mini([[1,2]]))
print(var([[1,2]]))
print(cen([[1,2,3,4,5,6,7,8,9]]))
print(std([[1,2]]))
print(rms([[1,2]]))
print(iqrr([[1,2,3,4,5,6,7]]))
print(mad([[1,2,3,4,5,9]]))
#print(zcr([data]))
#print(cec([data]))
print("cor:", cor([data, data]))
print(aco([pd.Series([0.968994140625, 0.9775390625, 0.971435546875])]))
'''