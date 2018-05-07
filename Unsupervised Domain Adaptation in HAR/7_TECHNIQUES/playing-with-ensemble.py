from EnsembleClassifier import EnsembleClassifier
#######################################
#######################################
#######################################
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import datasets
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

WINDOW = '3000'
train_pos = 'chest'
test_pos = train_pos
#
train = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 01_' + train_pos + '.csv')
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 02_' + train_pos + '.csv'))
train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 03_' + train_pos + '.csv'))
#train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 04_' + train_pos + '.csv'))
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
#train = train.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
# remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
train1 = train[train['label'].isin(['Walking'])]
train1 = train1.append(train[train['label'].isin(['Sitting'])])
#train1 = train1.append(train[train['label'].isin(['Bending'])])
#train1 = train1.append(train[train['label'].isin(['Hopping'])])
#train1 = train1.append(train[train['label'].isin(['Jogging'])])
#train1 = train1.append(train[train['label'].isin(['GoUpstairs'])])
train = train1
#
trainX = train.drop('label', 1)
trainY = train['label']
#
test = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 04_'+ train_pos +'.csv')
test = test.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
# remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
test1 = test[test['label'].isin(['Walking'])]
test1 = test1.append(test[test['label'].isin(['Sitting'])])
#test1 = test1.append(test[test['label'].isin(['Bending'])])
#test1 = test1.append(test[test['label'].isin(['Hopping'])])
#test1 = test1.append(test[test['label'].isin(['Jogging'])])
#test1 = test1.append(test[test['label'].isin(['GoDownstairs'])])
#test1 = test1.append(test[test['label'].isin(['GoUpstairs'])])
test = test1
#
testX = test.drop('label', 1)
testY = test['label']
# NORMALIZE
#print(trainX.describe())
#trainX = normalize_dataset(trainX)
#testX = normalize_dataset(testX)
#print(trainX.describe())
# BALANCE DATA
#trainX, trainY = balance_dataset(trainX, trainY)
#
def checkAccuracy(result, testY):
    p = 0
    for i, v in enumerate(result):
        if v == testY[i]:
            p += 1
    acc = p * 100 / len(result)
    # print(result)
    #print("ACC:{0}%, Total:{1}/{2} with positive {3}".format(acc, len(result), len(testY), p))
    return acc #, check_accuracy(result, testY)

########################
#### WITHOUT TL ########
########################
# Decision Tree
#print("\n Subspace Alignment (Fernando et al., 2013) ")
classifier_SA = SubspaceAlignedClassifier(loss="dtree")
classifier_SA.fit(trainX, trainY, testX)
pred_naive = classifier_SA.predict(testX)
acc_DT_SA = checkAccuracy(testY, pred_naive)
prob = classifier_SA.predict_proba(testX)
#print(prob)
print("acc_DT_SA:", acc_DT_SA);
# Logistic Regression
#print("\n Subspace Alignment (Fernando et al., 2013) ")
classifier = SubspaceAlignedClassifier(loss="logistic")
classifier.fit(trainX, trainY, testX)
pred_naive = classifier.predict(testX)
prob = classifier.predict_proba(testX)
#print(prob)
acc_LR_SA = checkAccuracy(testY, pred_naive)
print("acc_LR_SA:", acc_LR_SA); 

#i = 0
#for pred_label in pred_LR:
#    print(pred_label, np.max(prob[i]))
#    i+=1

#.predict(testX)
#acc_LR_SA = checkAccuracy(testY, pred)
#print("ACC:", acc_LR_SA); 
#
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
eclf = EnsembleClassifier(clfs=[ classifier_SA_DT, classifier_SA_LR, classifier_SA_NB,
    classifier_TCA_LR, classifier_TCA_DT, classifier_TCA_NB,
    classifier_NN_DT, classifier_NN_LR, classifier_NN_NB,
    classifier_KMM_DT, classifier_KMM_LR, classifier_KMM_NB ], weights=[1,1])

eclf.fit(trainX, trainY, testX)


pred = eclf.predict_v2(testX)
acc_ENSEMBLE_SA = checkAccuracy(testY, pred)
print("acc_ENSEMBLE_SA:", acc_ENSEMBLE_SA); 
#print(pred)

'''
#print(pred)
#
#print(checkAccuracy(testY, pred))
#

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[
         ('lr', clf1), ('rf', clf2), ('gnb', classifier_SA)], voting='hard')
eclf1 = eclf1.fit(trainX, trainY)
print(eclf1.score(testX, testY))
'''


