#### Experiments 
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.tca import TransferComponentClassifier
from libtlda.suba import SubspaceAlignedClassifier
from libtlda.scl import StructuralCorrespondenceClassifier
from libtlda.rba import RobustBiasAwareClassifier
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
from libtlda.tcpr import TargetContrastivePessimisticClassifier
# https://github.com/wmkouw/libTLDA

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
###
### smote, http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/over-sampling/plot_smote.html
###
from collections import Counter
from imblearn.over_sampling import SMOTE 
#
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
#
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
# ENSEMBLE
from EnsembleClassifier import EnsembleClassifier
#


def checkAccuracy(result, testY):
    p = 0
    for i, v in enumerate(result):
        if v == testY[i]:
            p += 1
    acc = p * 100 / len(result)
    # print(result)
    #print("ACC:{0}%, Total:{1}/{2} with positive {3}".format(acc, len(result), len(testY), p))
    return acc, check_accuracy(result, testY)


# without labels, only with numerical columns
def normalize_dataset(dataset):
    out = pd.DataFrame()
    x = dataset.copy()
    for i in x.columns:
        #print(i)
        new_col = normalize([x[i]], axis=1, norm='l2').ravel()
        out['norm_' + i] = new_col
    return out

def balance_dataset(X, Y):
    #print('Original dataset shape {}'.format(Counter(Y)))
    #Original dataset shape Counter({1: 900, 0: 100})
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, Y)
    #print('Resampled dataset shape {}'.format(Counter(y_res)))
    #Resampled dataset shape Counter({0: 900, 1: 900})
    return X_res, y_res

def check_accuracy(true, pred):
    repo = (classification_report(true, pred))
    labels = pred + true
    #print("len(labels):", len(set(labels)))
    total = len(set(labels))
    #print(repo)
    array = repo.split('\n')
    i = 0
    out2 = pd.DataFrame()
    out2_text = str()         
    #return out2
    #for col in out2.columns:
    #    out2_text += col + ":" + out2[col][0] + "\n"
    return repo

def read_file(path):
    try:
        content = pd.read_csv(path)
    except:
        content = pd.DataFrame();
    return content



WINDOW = '3000'
train_pos = 'chest'
test_pos = train_pos
#
train = pd.read_csv('C:\\Users\\paulo\\Documents\\py-har\\Unsupervised Domain Adaptation in HAR\\ANSAMO DATASET\\window_'+WINDOW+'ms_Subject 05_' + train_pos + '.csv')
train = train.append(pd.read_csv('C:\\Users\\paulo\\Documents\\py-har\\Unsupervised Domain Adaptation in HAR\\ANSAMO DATASET\\window_'+WINDOW+'ms_Subject 01_' + train_pos + '.csv'))

train1 = train[train['label'].isin(['Walking'])]
train1 = train1.append(train[train['label'].isin(['Sitting'])])
train1 = train1.append(train[train['label'].isin(['Bending'])])
train1 = train1.append(train[train['label'].isin(['Hopping'])])
train1 = train1.append(train[train['label'].isin(['Jogging'])])
train1 = train1.append(train[train['label'].isin(['GoUpstairs'])])
train = train1
#
trainX = train.drop('label', 1)
trainY = train['label']
#
test = pd.read_csv('C:\\Users\\paulo\\Documents\\py-har\\Unsupervised Domain Adaptation in HAR\\ANSAMO DATASET\\window_'+WINDOW+'ms_Subject 04_'+ train_pos +'.csv')
#test = test.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
# remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
test1 = test[test['label'].isin(['Walking'])]
test1 = test1.append(test[test['label'].isin(['Sitting'])])
test1 = test1.append(test[test['label'].isin(['Bending'])])
test1 = test1.append(test[test['label'].isin(['Hopping'])])
test1 = test1.append(test[test['label'].isin(['Jogging'])])
test1 = test1.append(test[test['label'].isin(['GoDownstairs'])])
test1 = test1.append(test[test['label'].isin(['GoUpstairs'])])
test = test1
#
testX = test.drop('label', 1)
testY = test['label']

# NORMALIZE
#print(trainX.describe())
trainX = normalize_dataset(trainX)
testX = normalize_dataset(testX)
#print(trainX.describe())
# BALANCE DATA
trainX, trainY = balance_dataset(trainX, trainY)
#
print("len(testY):", len(testY))

classifier_SA_DT = SubspaceAlignedClassifier(loss="dtree")
classifier_SA_LR = SubspaceAlignedClassifier(loss="logistic")

classifier_SA_NB = SubspaceAlignedClassifier(loss="berno")
classifier_TCA_DT = TransferComponentClassifier(loss="dtree")
classifier_TCA_LR = TransferComponentClassifier(loss="logistic")
classifier_TCA_NB = TransferComponentClassifier(loss="berno")
classifier_NN_DT = ImportanceWeightedClassifier(iwe='nn', loss="dtree")
classifier_NN_LR = ImportanceWeightedClassifier(iwe='nn', loss="logistic")
classifier_NN_NB = ImportanceWeightedClassifier(iwe='nn', loss="berno")
classifier_KMM_DT = ImportanceWeightedClassifier(iwe='kmm', loss="dtree")
classifier_KMM_LR = ImportanceWeightedClassifier(iwe='kmm', loss="logistic")
classifier_KMM_NB = ImportanceWeightedClassifier(iwe='kmm', loss="berno")
#
from EnsembleClassifier import EnsembleClassifier

eclf = EnsembleClassifier(clfs=[ 
    classifier_SA_DT, classifier_SA_LR, classifier_SA_NB,
    classifier_TCA_LR, classifier_TCA_DT, classifier_TCA_NB,
    classifier_NN_DT, classifier_NN_LR, classifier_NN_NB,
    classifier_KMM_DT, classifier_KMM_LR, classifier_KMM_NB])

eclf.fit(trainX, trainY, testX)
pred = eclf.predict(testX)
acc_ENSEMBLE, acc_ENSEMBLE_INFO = checkAccuracy(testY, pred)
print("acc_ENSEMBLE:", acc_ENSEMBLE); 