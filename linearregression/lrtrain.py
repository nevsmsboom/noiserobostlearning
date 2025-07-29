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



def getonedata(filename):
    nparray = np.load(filename)
    oldnparray = nparray - nparray.min()
    oldnparray.sort()
    

    newnparray = oldnparray[0:int(0.999*len(nparray))]  

    
    newnparraymaxvalue = newnparray.max()

    bins = [i*0.0002*newnparraymaxvalue for i in range(5001)]
    bins = bins 

    hist, _ = np.histogram(newnparray, bins = bins) 
    hist = hist / len(newnparray)
    histlist = hist.tolist() 
    
    i = filename
    a,b,c = i[:-4].split('_')[0], i[:-4].split('_')[1], i[:-4].split('_')[2] #c是标签，noiserate目标值，ab没用
    
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

    histlist.append(len(newnparray))
    histlist.append(num_classes)
    X = np.array(histlist)

    y = float(c)

    return X, y



task = 'VesselMNIST3D'

allfiles = os.listdir('alldata')
trainlist = []
testlist = []
for i in allfiles:
    if task in i:
        testlist.append(i)
    else: 
        trainlist.append(i)
        
        
X_train = []
y_train = []
for i in trainlist:
    X, y = getonedata('alldata/'+i)
    X_train.append(X)
    y_train.append(y)

X_test = []
y_test = []
for i in testlist:
    X, y = getonedata('alldata/'+i)
    X_test.append(X)
    y_test.append(y)




model = LinearRegression()  


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(mean_squared_error(y_pred, y_test))
print(mean_squared_error(0.2*np.ones([len(y_pred)]), y_test)) #与固定的noise rate对比

joblib.dump(model, task)


















'''

print(y_pred)

nparray=np.load('alldata/20231010loss_OrganMNIST3D_0.2.npy')
oldnparray = nparray - nparray.min()
oldnparray.sort()

#试试看去除一些最大值有没有影响？
newnparray = oldnparray[0:int(0.998*len(nparray))]  #目前看好像这个结果最好啊？去掉千分之一。。。惊呆
#newnparray = oldnparray[0:int(0.99*len(nparray))]
#newnparray = oldnparray

newnparraymaxvalue = newnparray.max()

bins = [i*0.0002*newnparraymaxvalue for i in range(5001)]

hist, _ = np.histogram(newnparray, bins = bins) #这里产生的hist是一个array，维度是类似(20,)这样
hist = hist / len(newnparray) #求比例
histlist = hist.tolist() #把array先变成list，方便后续append
histlist.append(len(newnparray))
histlist.append(11)
X_test2 = np.array(histlist).reshape(1,-1)

print(model.predict(X_test2))

'''