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
print("\n Kernel Mean Matching (Huang et al., 2006) ")
#classifier = ImportanceWeightedClassifier(iwe='kmm', loss="dtree")
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);

### Nearest-neighbour-based weighting (Loog, 2015)  
print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
#classifier = ImportanceWeightedClassifier(iwe='nn', loss="dtree")
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);


### Transfer Component Analysis (Pan et al, 2009) 
print("\n Transfer Component Analysis (Pan et al, 2009)")
#classifier = TransferComponentClassifier(loss="dtree")
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc)


### Subspace Alignment (Fernando et al., 2013) 
print("\n Subspace Alignment (Fernando et al., 2013) ")
#classifier = SubspaceAlignedClassifier(loss="dtree")
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);

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

### TargetContrastivePessimisticClassifier
#print("\n  TargetContrastivePessimisticClassifier    ")
#classifier = TargetContrastivePessimisticClassifier()
#classifier.fit(trainX, trainY, testX)
#pred_naive = classifier.predict(testX)
#acc = checkAccuracy(testY, pred_naive)
#print("ACC:", acc);


def build_models(trainX, trainY, testX, testY, source_pos, target_pos, window):
    #######################
    ### SEMI-SUPERVISED ###
    ########################
    # Label Propagation
    label_prop_model = LabelPropagation(kernel='knn')
    label_prop_model.fit(trainX, trainY)
    Y_Pred = label_prop_model.predict(testX);
    acc_ss_propagation, acc_ss_propagation_INFO = checkAccuracy(testY, Y_Pred)
    # Label Spreading
    label_prop_models_spr = LabelSpreading(kernel='knn')
    label_prop_models_spr.fit(trainX, trainY)
    Y_Pred = label_prop_models_spr.predict(testX);
    acc_ss_spreading, acc_ss_spreading_INFO = checkAccuracy(testY, Y_Pred)
    ########################
    #### WITHOUT TL ########
    ########################
    # LogisticRegression 
    modelLR = LogisticRegression()
    modelLR.fit(trainX, trainY)
    predLR = modelLR.predict(testX)
    accLR, acc_LR_INFO = checkAccuracy(testY, predLR)
    # DecisionTreeClassifier
    modelDT = tree.DecisionTreeClassifier()
    modelDT.fit(trainX, trainY)
    predDT = modelDT.predict(testX)
    accDT, acc_DT_INFO = checkAccuracy(testY, predDT)
    # BernoulliNB
    modelNB = BernoulliNB()
    modelNB.fit(trainX, trainY)
    predND = modelNB.predict(testX)
    accNB, acc_NB_INFO = checkAccuracy(testY, predND)
    #
    print("WITHOUT TL ACC_LR:", accLR, " ACC_DT:", accDT, " ACC_NB:", accNB)
    ########################
    #### WITH TL ########
    ########################

    ####################################################
    ### Kernel Mean Matching (Huang et al., 2006) 
    ###
    # Decision Tree
    print("\n Kernel Mean Matching (Huang et al., 2006) ")
    classifier = ImportanceWeightedClassifier(iwe='kmm', loss="dtree")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_KMM, acc_DT_KMM_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_DT_KMM)
    # Logistic Regression
    classifier = ImportanceWeightedClassifier(iwe='kmm', loss="logistic")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_KMM, acc_LR_KMM_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_LR_KMM);
    # Naive Bayes Bernoulli
    classifier = ImportanceWeightedClassifier(iwe='kmm', loss="berno")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_KMM, acc_NB_KMM_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_NB_KMM);
    ####################################################
    ### Nearest-neighbour-based weighting (Loog, 2015)  
    ###
    # Decision Tree
    print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
    classifier = ImportanceWeightedClassifier(iwe='nn', loss="dtree")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_NN, acc_DT_NN_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_DT_NN)
    # Logistic Regression
    print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
    classifier = ImportanceWeightedClassifier(iwe='nn', loss="logistic")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_NN, acc_LR_NN_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_LR_NN)        
    # Naive Bayes Bernoulli
    print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
    classifier = ImportanceWeightedClassifier(iwe='nn', loss="berno")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_NN, acc_NB_NN_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_NB_NN)     

    ####################################################
    ### Transfer Component Analysis (Pan et al, 2009) 
    ###
    # Decision Tree
    print("\n Transfer Component Analysis (Pan et al, 2009)")
    classifier = TransferComponentClassifier(loss="dtree", num_components=6)
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_TCA, acc_DT_TCA_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_DT_TCA)
    # Logistic Regression
    classifier = TransferComponentClassifier(loss="logistic", num_components=6)
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_TCA, acc_LR_TCA_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_LR_TCA)
    # Naive Bayes Bernoulli
    classifier = TransferComponentClassifier(loss="berno", num_components=6)
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_TCA, acc_NB_TCA_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_NB_TCA)

    ####################################################
    ### Subspace Alignment (Fernando et al., 2013) 
    ###
    # Decision Tree
    print("\n Subspace Alignment (Fernando et al., 2013) ")
    classifier = SubspaceAlignedClassifier(loss="dtree")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_SA, acc_DT_SA_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_DT_SA);
    # Logistic Regression
    print("\n Subspace Alignment (Fernando et al., 2013) ")
    classifier = SubspaceAlignedClassifier(loss="logistic")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_SA, acc_LR_SA_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_LR_SA);        
    # Naive Bayes Bernoulli        
    print("\n Subspace Alignment (Fernando et al., 2013) ")
    classifier = SubspaceAlignedClassifier(loss="berno")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_SA, acc_NB_SA_INFO = checkAccuracy(testY, pred_naive)
    print("ACC:", acc_NB_SA); 
    #################################
    ############# ENSEMBLE ##########
    #################################
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
    eclf = EnsembleClassifier(clfs=[ 
        classifier_TCA_DT,
        classifier_NN_DT,
        classifier_KMM_DT ])
    eclf.fit(trainX, trainY, testX)
    pred = eclf.predict_v2(testX)
    acc_ENSEMBLE, acc_ENSEMBLE_INFO = checkAccuracy(testY, pred)

    ########################
    #### RETURN ########
    ########################
    return pd.DataFrame(
        [{ 
        'window': window,
        'source_position': source_pos,
        'target_position': target_pos,

        'acc_SS_propagation': acc_ss_propagation,
        'acc_SS_propagation_INFO':acc_ss_propagation_INFO,
        'acc_SS_spreading': acc_ss_spreading,
        'acc_SS_spreading_INFO':acc_ss_spreading_INFO,

        'acc_ENSEMBLE': acc_ENSEMBLE,

        'acc_LR':accLR,
        'acc_LR_INFO': str(acc_LR_INFO),
        'acc_DT': accDT,
        'acc_DT_INFO': str(acc_DT_INFO),
        'acc_NB': accNB,
        'acc_NB_INFO': str(acc_NB_INFO),

        'acc_LR_KMM': acc_LR_KMM,
        'acc_LR_KMM_INFO': str(acc_LR_KMM_INFO),
        'acc_LR_NN': acc_LR_NN,
        'acc_LR_NN_INFO': str(acc_LR_NN_INFO),                
        'acc_LR_TCA': acc_LR_TCA,
        'acc_LR_TCA_INFO': str(acc_LR_TCA_INFO),                
        'acc_LR_SA': acc_LR_SA,
        'acc_LR_SA_INFO': str(acc_LR_SA_INFO),                

        'acc_DT_KMM': acc_DT_KMM,
        'acc_DT_KMM_INFO': str(acc_DT_KMM_INFO),                                
        'acc_DT_NN': acc_DT_NN,
        'acc_DT_NN_INFO': str(acc_DT_NN_INFO),                                
        'acc_DT_TCA': acc_DT_TCA,
        'acc_DT_TCA_INFO': str(acc_DT_TCA_INFO),                                
        'acc_DT_SA': acc_DT_SA,
        'acc_DT_SA_INFO': str(acc_DT_SA_INFO),                                
        
        'acc_NB_KMM': acc_NB_KMM,
        'acc_NB_KMM_INFO': str(acc_NB_KMM_INFO),                                
        'acc_NB_NN': acc_NB_NN,
        'acc_NB_NN_INFO': str(acc_NB_NN_INFO),                                                
        'acc_NB_TCA': acc_NB_TCA,
        'acc_NB_TCA_INFO': str(acc_NB_TCA_INFO),                                                
        'acc_NB_SA': acc_NB_SA,
        'acc_NB_SA_INFO': str(acc_NB_SA_INFO)                                            
        }]
    );


def ANSAMO_AGE():
    ########################
    #### LOAD DATA #########
    ########################
    positions = ['ankle', 'wrist', 'chest']    
    WINDOWS = ['2500', '3000']
    file_content = pd.DataFrame();
    for w in WINDOWS:
        WINDOW = w
        for position in positions:
            #   
            train_pos = position
            test_pos = position
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
            labels = ['Walking', 'Sitting', 'Hopping', 'Jogging']
            #         
            train = train[train['label'].isin(labels)]
            #
            trainX = train.drop('label', 1)
            trainY = train['label']
            test = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 07_'+ train_pos +'.csv')
            test = test.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 10_' + train_pos + '.csv'))
            # remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
            test1 = test[test['label'].isin(labels)]
            test = test1
            #
            testX = test.drop('label', 1)
            testY = test['label']
            # NORMALIZE
            trainX = normalize_dataset(trainX)
            testX = normalize_dataset(testX)
            # BALANCE DATA
            trainX, trainY = balance_dataset(trainX, trainY)
            #testX, testY = balance_dataset(testX, testY)
            #
            print("Normalizing and Balancing Data")
            #
            report = build_models(trainX, trainY, testX, testY, train_pos, test_pos, WINDOW);
            ########################
            #### WRITE TO FILE ########
            ########################
            file_content = file_content.append(report)
    file_content.to_csv('AGE-ANSAMO.csv', sep=';');



def ANSAMO_POS():
    ########################
    #### LOAD DATA #########
    ########################
    positions = ['ankle', 'wrist', 'chest', 'waist']    
    WINDOWS = ['2500', '3000']
    file_content = pd.DataFrame();
    for w in WINDOWS:
        WINDOW = w
        for train_pos in positions:
            for test_pos in positions:
                if train_pos is test_pos: 
                    continue
                #
                train = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 01_' + train_pos + '.csv')                 
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 02_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 03_' + train_pos + '.csv'))
                #train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 04_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 05_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 06_' + train_pos + '.csv'))
                #train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 07_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 08_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 09_' + train_pos + '.csv'))
                #train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 10_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 11_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 12_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 13_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 14_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 15_' + train_pos + '.csv'))
                #train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 16_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
                # remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
                labels = ['Walking', 'Sitting', 'Hopping', 'Jogging']
                train = train[train['label'].isin(labels)]
                #
                trainX = train.drop('label', 1)
                trainY = train['label']
                test = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 04_'+ test_pos +'.csv')
                test = test.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 16_' + train_pos + '.csv'))
                test = test[test['label'].isin(labels)]
                #
                testX = test.drop('label', 1)
                testY = test['label']
                # NORMALIZE
                trainX = normalize_dataset(trainX)
                testX = normalize_dataset(testX)
                # BALANCE DATA
                trainX, trainY = balance_dataset(trainX, trainY)
                testX, testY = balance_dataset(testX, testY)

                print("Normalizing and Balancing Data")
                #
                report = build_models(trainX, trainY, testX, testY, train_pos, test_pos, WINDOW);
                ########################
                #### WRITE TO FILE ########
                ########################
                file_content = file_content.append(report)

    file_content.to_csv('POS-ANSAMO.csv', sep=';');


ANSAMO_AGE()
#ANSAMO_POS()

#print(checkAccuracy(['A', 'A', 'A','A'], ['A', 'A', 'C', 'D']))
# print(cc)
#print(out)
#print(out2)

'''
#### Experiments 
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
train1 = train1.append(train[train['label'].isin(['Bending'])])
#train1 = train1.append(train[train['label'].isin(['Hopping'])])
#train1 = train1.append(train[train['label'].isin(['Jogging'])])
#train1 = train1.append(train[train['label'].isin(['GoUpstairs'])])
train = train1
#
trainX = train.drop('label', 1)
trainY = train['label']
#
test = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 04_'+ train_pos +'.csv')
#test = test.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
# remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
test1 = test[test['label'].isin(['Walking'])]
test1 = test1.append(test[test['label'].isin(['Sitting'])])
test1 = test1.append(test[test['label'].isin(['Bending'])])
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
trainX = normalize_dataset(trainX)
testX = normalize_dataset(testX)
#print(trainX.describe())
# BALANCE DATA
trainX, trainY = balance_dataset(trainX, trainY)
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
from EnsembleClassifier import EnsembleClassifier

eclf = EnsembleClassifier(clfs=[ 
    classifier_SA_DT, classifier_SA_LR, classifier_SA_NB,
    classifier_TCA_LR, classifier_TCA_DT, classifier_TCA_NB,
    classifier_NN_DT, classifier_NN_LR, classifier_NN_NB,
    classifier_KMM_DT, classifier_KMM_LR, classifier_KMM_NB ])
eclf.fit(trainX, trainY, testX)
pred = eclf.predict_v2(testX)
acc_ENSEMBLE, acc_ENSEMBLE_INFO = checkAccuracy(testY, pred)
print("acc_ENSEMBLE_SA:", acc_ENSEMBLE); 
'''