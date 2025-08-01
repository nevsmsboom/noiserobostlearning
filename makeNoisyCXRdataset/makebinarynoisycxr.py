
import os
import pandas as pd
import json
import numpy as np
import joblib
import random


with open('filesneeded/pneumonia-challenge-dataset-mappings_2018.json','r', encoding='gbk', ) as js:
    mappingpairs = json.load(js)
    





#kaggle(clean lables)
kagglefilelist = []
kagglelabels = []
kaggledf = pd.read_csv('Kaggleversion/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
for k in range(len(kaggledf)-1):
    if kaggledf.loc[k]['patientId'] != kaggledf.loc[k+1]['patientId']:
        #if kaggledf.loc[k]['x'] >= 0:
        kagglelabels.append(kaggledf.loc[k]['Target'])
        kagglefilelist.append(kaggledf.loc[k]['patientId'])
    else:
        pass 
k = len(kaggledf)-1
kagglelabels.append(kaggledf.loc[k]['Target'])
kagglefilelist.append(kaggledf.loc[k]['patientId'])

cleanlabels = kagglelabels







#NIH labels
nihfilelist = []
nihlabels = []

for i in kagglefilelist:
    for j in mappingpairs:
        if i == j['subset_img_id']:
            #print(i)
            nihfilelist.append(j['img_id'])
            nihlabels.append(j['orig_labels'])
            break


#noise labels
noisylabels = []
for ll in range(len(nihlabels)):
    if 'Pneumonia' in nihlabels[ll]:
        noisylabels.append(1)
        
        
    elif 'Infiltration' in nihlabels[ll] or 'Consolidation' in nihlabels[ll]:
        randomdice = random.random()
        if randomdice >= 0.30:
            noisylabels.append(cleanlabels[ll])
        else:
            noisylabels.append(1-cleanlabels[ll])
        
    #elif 'No Finding' in nihlabels[ll]:
    #    noisylabels.append(0)
        
    else:
        noisylabels.append(0)






#查看noiserate
cleanarray = np.array(cleanlabels)
noisyarray = np.array(noisylabels)
equal = cleanarray==noisyarray
print('noiserate', equal.sum()/len(cleanarray))
print('datadistribution', (26684-noisyarray.sum())/26684)

unique_values1, counts1 = np.unique(cleanarray, return_counts=True)
for value, count in zip(unique_values1, counts1):
    print('In clean array,', f"Value {value} appears {count} times.")
unique_values, counts = np.unique(noisyarray, return_counts=True)
for value, count in zip(unique_values, counts):
    print('In noisy array,', f"Value {value} appears {count} times.")





finaldict = dict()
finaldict['kagglefilelist'] = kagglefilelist
finaldict['nihfilelist'] = nihfilelist
#finaldict['nihlabels'] = nihlabels
#finaldict['kagglelabels'] = kagglelabels
finaldict['noisylabels'] = noisylabels 
finaldict['cleanlabels'] = cleanlabels 


joblib.dump(finaldict, 'dictfordataset')

