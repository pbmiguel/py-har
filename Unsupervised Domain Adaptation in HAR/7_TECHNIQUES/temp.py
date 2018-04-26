from libtlda.iw import ImportanceWeightedClassifier
from libtlda.tca import TransferComponentClassifier
from libtlda.suba import SubspaceAlignedClassifier
from libtlda.scl import StructuralCorrespondenceClassifier
from libtlda.rba import RobustBiasAwareClassifier
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
from libtlda.tcpr import TargetContrastivePessimisticClassifier
# https://github.com/wmkouw/libTLDA

import pandas as pd
import numpy as np

def checkAccuracy(result, testY):
    p = 0
    for i, v in enumerate(result):
        if v == testY[i]:
            p += 1
    acc = p * 100 / len(result)
    # print(result)
    print("ACC:{0}%, Total:{1}/{2} with positive {3}".format(acc, len(result), len(testY), p))
    return acc

positions = ['ankle', 'wrist', 'chest']    
WINDOW = '3000'
train_pos = positions[0]
test_pos = positions[0]
#
train = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 01_' + train_pos + '.csv')
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 02_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 03_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 04_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 05_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 06_' + train_pos + '.csv'))
#train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 07_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 08_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 09_' + train_pos + '.csv'))
#train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 10_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 11_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 12_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 13_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 14_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 15_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 16_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
# remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
train1 = train[train['label'].isin(['Walking'])]
train1 = train1.append(train[train['label'].isin(['Sitting'])])
train1 = train1.append(train[train['label'].isin(['Bending'])])
train1 = train1.append(train[train['label'].isin(['Hopping'])])
train = train1
#
trainX = train.drop('label', 1)
trainY = train['label']
test = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 07_'+ train_pos +'.csv')
test = test.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 10_' + train_pos + '.csv'))
# remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
test1 = test[test['label'].isin(['Walking'])]
test1 = test1.append(test[test['label'].isin(['Sitting'])])
test1 = test1.append(test[test['label'].isin(['Bending'])])
test1 = test1.append(test[test['label'].isin(['Hopping'])])
test = test1
#
testX = test.drop('label', 1)
testY = test['label']
#
#
#
'''
Importance-weighted classifier, with weight estimators:
    1. Kernel density estimation 
    2. Ratio of Gaussians (Shimodaira, 2000) 
    3. Logistic discrimination (Bickel et al., 2009) 
    4. Kernel Mean Matching (Huang et al., 2006) 
    5. Nearest-neighbour-based weighting (Loog, 2015) 
6. Transfer Component Analysis (Pan et al, 2009) 
7. Subspace Alignment (Fernando et al., 2013) 
8. Structural Correspondence Learning (Blitzer et al., 2006) 
9. Robust Bias-Aware (Liu & Ziebart, 2014) 
10. Feature-Level Domain Adaptation (Kouw et al., 2016) 
'''


### Kernel density estimation 
#print("\nKernel density estimation")
#classifier = ImportanceWeightedClassifier(iwe='kde')
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);

### Ratio of Gaussians (Shimodaira, 2000)  
#print("\n Ratio of Gaussians (Shimodaira, 2000) ")
#classifier = ImportanceWeightedClassifier(iwe='rg')
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);

### Logistic discrimination (Bickel et al., 2009) 
#print("\n Logistic discrimination (Bickel et al., 2009)  ")
#classifier = ImportanceWeightedClassifier(iwe='rg')
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);


### Kernel Mean Matching (Huang et al., 2006) 
print("\n Kernel Mean Matching (Huang et al., 2006)   ")
classifier = ImportanceWeightedClassifier(iwe='kmm')
classifier.fit(trainX, trainY, testX)
pred_naive = classifier.predict(testX)
acc = checkAccuracy(testY, pred_naive)
print("ACC:", acc);

### Nearest-neighbour-based weighting (Loog, 2015)  
print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
classifier = ImportanceWeightedClassifier(iwe='nn')
classifier.fit(trainX, trainY, testX)
pred_naive = classifier.predict(testX)
acc = checkAccuracy(testY, pred_naive)
print("ACC:", acc);


### Transfer Component Analysis (Pan et al, 2009) 
print("\nTransfer Component Analysis (Pan et al, 2009)")
classifier = TransferComponentClassifier()
classifier.fit(trainX, trainY, testX)
pred_naive = classifier.predict(testX)
acc = checkAccuracy(testY, pred_naive)
print("ACC:", acc);

### Subspace Alignment (Fernando et al., 2013) 
print("\n Subspace Alignment (Fernando et al., 2013) ")
classifier = SubspaceAlignedClassifier()
classifier.fit(trainX, trainY, testX)
pred_naive = classifier.predict(testX)
acc = checkAccuracy(testY, pred_naive)
print("ACC:", acc);

### Structural Correspondence Learning (Blitzer et al., 2006)  
#print("\n Structural Correspondence Learning (Blitzer et al., 2006)  ")
#classifier = StructuralCorrespondenceClassifier()
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);

### Robust Bias-Aware (Liu & Ziebart, 2014)  
#print("\n Robust Bias-Aware (Liu & Ziebart, 2014)   ")
#classifier = RobustBiasAwareClassifier()
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);

### R Feature-Level Domain Adaptation (Kouw et al., 2016) 
#print("\n  Feature-Level Domain Adaptation (Kouw et al., 2016)    ")
#classifier = FeatureLevelDomainAdaptiveClassifier()
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);
