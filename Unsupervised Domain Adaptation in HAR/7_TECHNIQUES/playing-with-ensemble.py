'''import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

eclf1 = VotingClassifier(estimators=[         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X, y)
print(eclf1.predict(X))

eclf2 = VotingClassifier(estimators=[         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],         voting='soft')
eclf2 = eclf2.fit(X, y)
print(eclf2.predict(X))

eclf3 = VotingClassifier(estimators=[
    ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft', weights=[2,1,1],
    flatten_transform=True)
eclf3 = eclf3.fit(X, y)

print(eclf3.predict(X))
print(eclf3.transform(X).shape)'''

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y, Z):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for clf in self.clfs:
            clf.fit(X, y, Z)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """

        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        #avg = np.average(self.probas_, axis=0, weights=self.weights)

        return self.probas_


    def majority_voting(self, proposed_prediction):
        final_predictions = [0 for y in range(len(proposed_prediction[0]))]
        #print(le)
        for i in range(len(proposed_prediction[0])):
            voting = dict()

            for col in range(len(proposed_prediction)):
                #print(i)
                label, prob = proposed_prediction[col][i]
                try:
                    values = voting[label] 
                    voting[label] = values+1
                except KeyError:
                    voting[label] = 1
                
                #final_predictions[i]
                #print(label, prob)

            #print("\n", voting)
            max = 0
            vote = ""
            for k in voting:
                if voting[k] > max:
                    max = voting[k]
                    vote = k     
            #print(len(proposed_prediction[0]))  
            #print(i)             
            final_predictions[i] = vote

        return final_predictions

    def predict_v2(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """
        algorithms =  self.clfs
        #x = range(0,1)
        #w, h = 8, 5;
        #Matrix = [[0 for x in range(w)] for y in range(h)] 
        #predictions = np.reshape(x, (len(X), len(algorithms)))

        predictions = [0 for y in range(len(algorithms))]
        probabilities = [0 for y in range(len(algorithms))]
        final_predictions = [0 for y in range(len(algorithms))]

        for col, alg in enumerate(algorithms):
            predictions[col] = alg.predict(X)
            probabilities[col] = alg.predict_proba(testX)
            pred_prob = [0 for y in range(len(X))]
            for i, row in enumerate(pred_prob):
                pred_prob[i] = (predictions[col][i], np.max(probabilities[col][i]))
            final_predictions[col] = pred_prob
        
        #print(final_predictions)
        return self.majority_voting(final_predictions)





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
#
eclf = EnsembleClassifier(clfs=[classifier_SA_DT, classifier_SA_LR], weights=[1,1])
eclf.fit(trainX, trainY, testX)
print("len(testX):", len(testX))
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


