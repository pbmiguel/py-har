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
# 
from utils import EnsembleClassifier
from utils import check_accuracy
from utils import normalize_dataset
from utils import balance_dataset
from utils import add_avg_to_report
#
from expAA1_tl import apply_ENSEMBLE
from expAA1_tl import apply_KMM
from expAA1_tl import apply_NN
from expAA1_tl import apply_SA
from expAA1_tl import apply_TCA

def apply_notl(trainX, trainY, testX, testY, window, source_pos, target_pos):
    #######################
    ### SEMI-SUPERVISED ###
    ########################
    # Label Propagation
    label_prop_model = LabelPropagation(kernel='knn')
    label_prop_model.fit(trainX, trainY)
    Y_Pred = label_prop_model.predict(testX);
    acc_ss_propagation, acc_ss_propagation_INFO = check_accuracy(testY, Y_Pred)
    # Label Spreading
    label_prop_models_spr = LabelSpreading(kernel='knn')
    label_prop_models_spr.fit(trainX, trainY)
    Y_Pred = label_prop_models_spr.predict(testX);
    acc_ss_spreading, acc_ss_spreading_INFO = check_accuracy(testY, Y_Pred)
    ########################
    #### WITHOUT TL ########
    ########################
    # LogisticRegression 
    modelLR = LogisticRegression()
    modelLR.fit(trainX, trainY)
    predLR = modelLR.predict(testX)
    accLR, acc_LR_INFO = check_accuracy(testY, predLR)
    # DecisionTreeClassifier
    modelDT = tree.DecisionTreeClassifier()
    modelDT.fit(trainX, trainY)
    predDT = modelDT.predict(testX)
    accDT, acc_DT_INFO = check_accuracy(testY, predDT)
    # BernoulliNB
    modelNB = BernoulliNB()
    modelNB.fit(trainX, trainY)
    predND = modelNB.predict(testX)
    accNB, acc_NB_INFO = check_accuracy(testY, predND)
    #
    return pd.DataFrame(
        [{ 
        'window': window,
        'source_position': source_pos,
        'target_position': target_pos,

        'acc_SS_propagation': acc_ss_propagation,
        'acc_SS_propagation_INFO':acc_ss_propagation_INFO,
        'acc_SS_spreading': acc_ss_spreading,
        'acc_SS_spreading_INFO':acc_ss_spreading_INFO,
        'acc_LR':accLR,
        'acc_LR_INFO': str(acc_LR_INFO),
        'acc_DT': accDT,
        'acc_DT_INFO': str(acc_DT_INFO),
        'acc_NB': accNB,
        'acc_NB_INFO': str(acc_NB_INFO)       

        }]
    )

def run_expAA1(OUTPUT_PATH):
    ########################
    #### LOAD DATA #########
    ########################
    positions = ['ankle', 'wrist', 'chest']    
    WINDOWS = ['2500', '3000']
    file_without_tl, file_with_tca, file_with_kmm, file_with_nn, file_with_sa, file_with_en  = pd.DataFrame(), pd.DataFrame(),pd.DataFrame(), pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    PATH = './datasets/ANSAMO/'
    #
    for w in WINDOWS:
        WINDOW = w
        for position in positions:
            #   
            train_pos = position
            test_pos = position
            #
            train = pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 01_' + train_pos + '.csv')
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 02_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 03_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 04_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 05_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 06_' + train_pos + '.csv'))
            #train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 07_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 08_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 09_' + train_pos + '.csv'))
            #train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 10_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 11_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 12_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 13_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 14_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 15_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 16_' + train_pos + '.csv'))
            train = train.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
            # remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
            labels = ['Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging']
            #         
            train = train[train['label'].isin(labels)]
            #
            trainX = train.drop('label', 1)
            trainY = train['label']
            test = pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 07_'+ train_pos +'.csv')
            test = test.append(pd.read_csv(PATH + 'window_'+WINDOW+'ms_Subject 10_' + train_pos + '.csv'))
            # remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
            test1 = test[test['label'].isin(labels)]
            test = test1
            #
            testX = test.drop('label', 1)
            testY = test['label']
            # NORMALIZE
            trainX = normalize_dataset(trainX, 'l2')
            testX = normalize_dataset(testX, 'l2')
            # BALANCE DATA
            trainX, trainY = balance_dataset(trainX, trainY)
            #testX, testY = balance_dataset(testX, testY)
            #
            print("Normalizing and Balancing Data")
            #
            ########################
            #### WRITE TO FILE ########
            ########################
            '''file_without_tl = file_without_tl.append(apply_notl(trainX, trainY, testX, testY, WINDOW, train_pos, test_pos))
            file_with_nn = file_with_nn.append(apply_NN(trainX, trainY, testX, testY, WINDOW, train_pos, test_pos))
            file_with_kmm = file_with_kmm.append(apply_KMM(trainX, trainY, testX, testY, WINDOW, train_pos, test_pos))
            file_with_sa = file_with_sa.append(apply_SA(trainX, trainY, testX, testY, WINDOW, train_pos, test_pos))'''
            file_with_tca = file_with_tca.append(apply_TCA(trainX, trainY, testX, testY, WINDOW, train_pos, test_pos))
            '''file_with_en = file_with_en.append(apply_ENSEMBLE(trainX, trainY, testX, testY, WINDOW, train_pos, test_pos))'''

    '''# without tl
    file_without_tl = add_avg_to_report(file_without_tl)
    file_without_tl.to_csv(OUTPUT_PATH + 'exp-AA1-wihout-tl.csv', sep=';');'''
    # tca
    file_with_tca = add_avg_to_report(file_with_tca)
    file_with_tca.to_csv(OUTPUT_PATH + 'exp-AA1-with-tca.csv', sep=';');
    '''# sa
    file_with_sa = add_avg_to_report(file_with_sa)
    file_with_sa.to_csv(OUTPUT_PATH + 'exp-AA1-with-sa.csv', sep=';');
    # kmm
    file_with_kmm = add_avg_to_report(file_with_kmm)
    file_with_kmm.to_csv(OUTPUT_PATH + 'exp-AA1-with-kmm.csv', sep=';');    
    # nn
    file_with_nn = add_avg_to_report(file_with_nn)
    file_with_nn.to_csv(OUTPUT_PATH + 'exp-AA1-with-nn.csv', sep=';');
    # en
    file_with_en = add_avg_to_report(file_with_en)
    file_with_en.to_csv(OUTPUT_PATH + 'exp-AA1-with-en.csv', sep=';');    '''


'''
Observations:
    - SUP best accuracy is 60%
    - SS best accuracy is 30%
    - KMM is good with DT and LR (41,30)
    - NN is good with DT (39)
    - SA is bad with everyone
    - TCA is good with NB (44)
'''