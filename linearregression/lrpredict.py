import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

#from sklearn.externals import joblib as sklearnjoblib
import joblib


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def lrpredict(linearregressionmodel, nparray, size_dataset, num_classes):
    
    
    oldnparray = nparray - nparray.min()
    oldnparray.sort()
    

    newnparray = oldnparray[0:int(0.999*len(nparray))]  

    
    newnparraymaxvalue = newnparray.max()

    bins = [i*0.0002*newnparraymaxvalue for i in range(5001)]

    hist, _ = np.histogram(newnparray, bins = bins) 
    hist = hist / len(newnparray)
    histlist = hist.tolist() 

    '''
    i = filename
    a,b,c = i[:-4].split('_')[0], i[:-4].split('_')[1], i[:-4].split('_')[2] 
    
    if 'Path' in i:
        num_classes = 9
    if 'OCT' in i:
        num_classes = 4
    if 'Pneumonia' in i:
        num_classes = 2
    if 'Organ' in i:
        num_classes = 11
    if 'Vessel' in i:
        num_classes = 2
    '''

    histlist.append(len(newnparray))
    histlist.append(num_classes)
    X_test = np.array(histlist).reshape(1,-1)

    #y = float(c)

    y_pred = linearregressionmodel.predict(X_test)

    return y_pred


'''
a = lrpredict(linearregressionmodel=joblib.load('OrganMNIST3D'), nparray=np.load('alldata/20231010loss_OrganMNIST3D_0.2.npy'), size_dataset=972, num_classes=11)
print(a)


'''