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

import pandas as pd
import numpy as np

def checkAccuracy(result, testY):
    p = 0
    for i, v in enumerate(result):
        if v == testY[i]:
            p += 1
    acc = p * 100 / len(result)
    # print(result)
    #print("ACC:{0}%, Total:{1}/{2} with positive {3}".format(acc, len(result), len(testY), p))
    return acc


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
            train1 = train[train['label'].isin(['Walking'])]
            train1 = train1.append(train[train['label'].isin(['Sitting'])])
            train1 = train1.append(train[train['label'].isin(['Bending'])])
            train1 = train1.append(train[train['label'].isin(['Hopping'])])
            train1 = train1.append(train[train['label'].isin(['GoDownstairs'])])
            train1 = train1.append(train[train['label'].isin(['GoUpstairs'])])
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
            test1 = test1.append(test[test['label'].isin(['GoDownstairs'])])
            test1 = test1.append(test[test['label'].isin(['GoUpstairs'])])
            test = test1
            #
            testX = test.drop('label', 1)
            testY = test['label']
            ########################
            #### WITHOUT TL ########
            ########################
            # LogisticRegression 
            modelLR = LogisticRegression()
            modelLR.fit(trainX, trainY)
            predLR = modelLR.predict(testX)
            accLR = checkAccuracy(testY, predLR)
            # DecisionTreeClassifier
            modelDT = tree.DecisionTreeClassifier()
            modelDT.fit(trainX, trainY)
            predDT = modelDT.predict(testX)
            accDT = checkAccuracy(testY, predDT)
            # BernoulliNB
            modelNB = BernoulliNB()
            modelNB.fit(trainX, trainY)
            predND = modelNB.predict(testX)
            accNB = checkAccuracy(testY, predND)
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
            acc_DT_KMM = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_DT_KMM)
            # Logistic Regression
            classifier = ImportanceWeightedClassifier(iwe='kmm', loss="logistic")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_LR_KMM = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_LR_KMM);
            # Naive Bayes Bernoulli
            classifier = ImportanceWeightedClassifier(iwe='kmm', loss="berno")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_NB_KMM = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_NB_KMM);
            ####################################################
            ### Nearest-neighbour-based weighting (Loog, 2015)  
            ###
            # Decision Tree
            print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
            classifier = ImportanceWeightedClassifier(iwe='nn', loss="dtree")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_DT_NN = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_DT_NN)
            # Logistic Regression
            print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
            classifier = ImportanceWeightedClassifier(iwe='nn', loss="logistic")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_LR_NN = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_LR_NN)        
            # Naive Bayes Bernoulli
            print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
            classifier = ImportanceWeightedClassifier(iwe='nn', loss="berno")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_NB_NN = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_NB_NN)     

            ####################################################
            ### Transfer Component Analysis (Pan et al, 2009) 
            ###
            # Decision Tree
            print("\n Transfer Component Analysis (Pan et al, 2009)")
            classifier = TransferComponentClassifier(loss="dtree")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_DT_TCA = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_DT_TCA)
            # Logistic Regression
            classifier = TransferComponentClassifier(loss="logistic")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_LR_TCA = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_LR_TCA)
            # Naive Bayes Bernoulli
            classifier = TransferComponentClassifier(loss="berno")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_NB_TCA = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_NB_TCA)

            ####################################################
            ### Subspace Alignment (Fernando et al., 2013) 
            ###
            # Decision Tree
            print("\n Subspace Alignment (Fernando et al., 2013) ")
            classifier = SubspaceAlignedClassifier(loss="dtree")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_DT_SA = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_DT_SA);
            # Logistic Regression
            print("\n Subspace Alignment (Fernando et al., 2013) ")
            classifier = SubspaceAlignedClassifier(loss="logistic")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_LR_SA = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_LR_SA);        
            # Naive Bayes Bernoulli        
            print("\n Subspace Alignment (Fernando et al., 2013) ")
            classifier = SubspaceAlignedClassifier(loss="berno")
            classifier.fit(trainX, trainY, testX)
            pred_naive = classifier.predict(testX)
            acc_NB_SA = checkAccuracy(testY, pred_naive)
            print("ACC:", acc_NB_SA);     
            
            ########################
            #### WRITE TO FILE ########
            ########################
            file_content = file_content.append(pd.DataFrame(
                [{ 
                'window': WINDOW,
                'train_position': train_pos,
                'test_position': test_pos,
                'acc_LR':accLR,
                'acc_DT': accDT,
                'acc_NB': accNB,
                
                'acc_LR_KMM': acc_LR_KMM,
                'acc_LR_NN': acc_LR_NN,
                'acc_LR_TCA': acc_LR_TCA,
                'acc_LR_SA': acc_LR_SA,

                'acc_DT_KMM': acc_DT_KMM,
                'acc_DT_NN': acc_DT_NN,
                'acc_DT_TCA': acc_DT_TCA,
                'acc_DT_SA': acc_DT_SA,
                
                'acc_NB_KMM': acc_NB_KMM,
                'acc_NB_NN': acc_NB_NN,
                'acc_NB_TCA': acc_NB_TCA,
                'acc_NB_SA': acc_NB_SA
                }]
            ));
    file_content.to_csv('AGE-ANSAMO.csv', sep=';');


def read_file(path):
    try:
        content = pd.read_csv(path)
    except:
        content = pd.DataFrame();
    return content

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
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 16_' + train_pos + '.csv'))
                train = train.append(read_file('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 17_' + train_pos + '.csv'))
                # remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
                train1 = train[train['label'].isin(['Walking'])]
                train1 = train1.append(train[train['label'].isin(['Sitting'])])
                train1 = train1.append(train[train['label'].isin(['Bending'])])
                train1 = train1.append(train[train['label'].isin(['Hopping'])])
                train1 = train1.append(train[train['label'].isin(['Jogging'])])
                #train1 = train1.append(train[train['label'].isin(['GoUpstairs'])])
                train = train1
                #
                trainX = train.drop('label', 1)
                trainY = train['label']
                test = pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 04_'+ test_pos +'.csv')
                #test = test.append(pd.read_csv('../ANSAMO DATASET/window_'+WINDOW+'ms_Subject 10_' + train_pos + '.csv'))
                # remove certain labels {'Bending', 'GoDownstairs', 'Hopping', 'Walking', 'Sitting', 'GoUpstairs', 'Jogging'}
                test1 = test[test['label'].isin(['Walking'])]
                test1 = test1.append(test[test['label'].isin(['Sitting'])])
                test1 = test1.append(test[test['label'].isin(['Bending'])])
                test1 = test1.append(test[test['label'].isin(['Hopping'])])
                test1 = test1.append(test[test['label'].isin(['Jogging'])])
                #test1 = test1.append(test[test['label'].isin(['GoUpstairs'])])
                test = test1
                #
                testX = test.drop('label', 1)
                testY = test['label']
                            ########################
                #### WITHOUT TL ########
                ########################
                # LogisticRegression 
                modelLR = LogisticRegression()
                modelLR.fit(trainX, trainY)
                predLR = modelLR.predict(testX)
                accLR = checkAccuracy(testY, predLR)
                # DecisionTreeClassifier
                modelDT = tree.DecisionTreeClassifier()
                modelDT.fit(trainX, trainY)
                predDT = modelDT.predict(testX)
                accDT = checkAccuracy(testY, predDT)
                # BernoulliNB
                modelNB = BernoulliNB()
                modelNB.fit(trainX, trainY)
                predND = modelNB.predict(testX)
                accNB = checkAccuracy(testY, predND)
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
                acc_DT_KMM = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_DT_KMM)
                # Logistic Regression
                classifier = ImportanceWeightedClassifier(iwe='kmm', loss="logistic")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_LR_KMM = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_LR_KMM);
                # Naive Bayes Bernoulli
                classifier = ImportanceWeightedClassifier(iwe='kmm', loss="berno")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_NB_KMM = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_NB_KMM);
                ####################################################
                ### Nearest-neighbour-based weighting (Loog, 2015)  
                ###
                # Decision Tree
                print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
                classifier = ImportanceWeightedClassifier(iwe='nn', loss="dtree")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_DT_NN = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_DT_NN)
                # Logistic Regression
                print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
                classifier = ImportanceWeightedClassifier(iwe='nn', loss="logistic")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_LR_NN = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_LR_NN)        
                # Naive Bayes Bernoulli
                print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
                classifier = ImportanceWeightedClassifier(iwe='nn', loss="berno")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_NB_NN = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_NB_NN)     

                ####################################################
                ### Transfer Component Analysis (Pan et al, 2009) 
                ###
                # Decision Tree
                print("\n Transfer Component Analysis (Pan et al, 2009)")
                classifier = TransferComponentClassifier(loss="dtree")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_DT_TCA = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_DT_TCA)
                # Logistic Regression
                classifier = TransferComponentClassifier(loss="logistic")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_LR_TCA = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_LR_TCA)
                # Naive Bayes Bernoulli
                classifier = TransferComponentClassifier(loss="berno")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_NB_TCA = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_NB_TCA)

                ####################################################
                ### Subspace Alignment (Fernando et al., 2013) 
                ###
                # Decision Tree
                print("\n Subspace Alignment (Fernando et al., 2013) ")
                classifier = SubspaceAlignedClassifier(loss="dtree")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_DT_SA = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_DT_SA);
                # Logistic Regression
                print("\n Subspace Alignment (Fernando et al., 2013) ")
                classifier = SubspaceAlignedClassifier(loss="logistic")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_LR_SA = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_LR_SA);        
                # Naive Bayes Bernoulli        
                print("\n Subspace Alignment (Fernando et al., 2013) ")
                classifier = SubspaceAlignedClassifier(loss="berno")
                classifier.fit(trainX, trainY, testX)
                pred_naive = classifier.predict(testX)
                acc_NB_SA = checkAccuracy(testY, pred_naive)
                print("ACC:", acc_NB_SA);     
                
                ########################
                #### WRITE TO FILE ########
                ########################
                file_content = file_content.append(pd.DataFrame(
                    [{ 
                    'window': WINDOW,
                    'train_position': train_pos,
                    'test_position': test_pos,
                    'acc_LR':accLR,
                    'acc_DT': accDT,
                    'acc_NB': accNB,
                    
                    'acc_LR_KMM': acc_LR_KMM,
                    'acc_LR_NN': acc_LR_NN,
                    'acc_LR_TCA': acc_LR_TCA,
                    'acc_LR_SA': acc_LR_SA,

                    'acc_DT_KMM': acc_DT_KMM,
                    'acc_DT_NN': acc_DT_NN,
                    'acc_DT_TCA': acc_DT_TCA,
                    'acc_DT_SA': acc_DT_SA,
                    
                    'acc_NB_KMM': acc_NB_KMM,
                    'acc_NB_NN': acc_NB_NN,
                    'acc_NB_TCA': acc_NB_TCA,
                    'acc_NB_SA': acc_NB_SA
                    }]
                ));
    file_content.to_csv('POS-ANSAMO.csv', sep=';');

ANSAMO_POS()