from libtlda.iw import ImportanceWeightedClassifier
from libtlda.tca import TransferComponentClassifier
from libtlda.suba import SubspaceAlignedClassifier
from libtlda.scl import StructuralCorrespondenceClassifier
from libtlda.rba import RobustBiasAwareClassifier
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
from libtlda.tcpr import TargetContrastivePessimisticClassifier
# https://github.com/wmkouw/libTLDA
from utils import EnsembleClassifier
from utils import check_accuracy
from utils import normalize_dataset
from utils import balance_dataset
from utils import add_avg_to_report
#
import pandas as pd


#################################
############# ENSEMBLE ##########
#################################
def apply_ENSEMBLE(trainX, trainY, testX, testY, window, source_pos, target_pos):
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
        #classifier_SA_DT,
        #classifier_SA_LR,
        #classifier_SA_NB,

        #classifier_TCA_DT,
        #classifier_TCA_LR,
        classifier_TCA_NB,

        classifier_NN_DT,
        #classifier_NN_LR,
        #classifier_NN_NB,

        classifier_KMM_DT,
        classifier_KMM_LR,
        #classifier_KMM_NB
         ])
    eclf.fit(trainX, trainY, testX)
    pred = eclf.predict(testX)
    acc_ENSEMBLE, acc_ENSEMBLE_INFO = check_accuracy(testY, pred)
    #
    return pd.DataFrame(
        [{ 
        'window': window,
        'source_position': source_pos,
        'target_position': target_pos,

        'acc_ENSEMBLE': acc_ENSEMBLE,  
        'acc_ENSEMBLE_INFO': acc_ENSEMBLE_INFO,                                            
        }]
    )

###
### Subspace Alignment (Fernando et al., 2013) 
###
def apply_SA(trainX, trainY, testX, testY, window, source_pos, target_pos):
    # Decision Tree
    print("\n Subspace Alignment (Fernando et al., 2013) ")
    classifier = SubspaceAlignedClassifier(loss="dtree")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_SA, acc_DT_SA_INFO = check_accuracy(testY, pred_naive)
    # Logistic Regression
    print("\n Subspace Alignment (Fernando et al., 2013) ")
    classifier = SubspaceAlignedClassifier(loss="logistic")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_SA, acc_LR_SA_INFO = check_accuracy(testY, pred_naive)
    # Naive Bayes Bernoulli        
    print("\n Subspace Alignment (Fernando et al., 2013) ")
    classifier = SubspaceAlignedClassifier(loss="berno")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_SA, acc_NB_SA_INFO = check_accuracy(testY, pred_naive)
    #
    return pd.DataFrame(
            [{ 
            'window': window,
            'source_position': source_pos,
            'target_position': target_pos,

            'acc_LR_SA': acc_LR_SA,
            'acc_LR_SA_INFO': str(acc_LR_SA_INFO),                                              
            'acc_DT_SA': acc_DT_SA,
            'acc_DT_SA_INFO': str(acc_DT_SA_INFO),                                                                            
            'acc_NB_SA': acc_NB_SA,
            'acc_NB_SA_INFO': str(acc_NB_SA_INFO),                                                
            }]
        )
###
### Kernel Mean Matching (Huang et al., 2006) 
###
def apply_KMM(trainX, trainY, testX, testY, window, source_pos, target_pos):
    # Decision Tree
    print("\n Kernel Mean Matching (Huang et al., 2006) ")
    classifier = ImportanceWeightedClassifier(iwe='kmm', loss="dtree")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_KMM, acc_DT_KMM_INFO = check_accuracy(testY, pred_naive)
    # Logistic Regression
    classifier = ImportanceWeightedClassifier(iwe='kmm', loss="logistic")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_KMM, acc_LR_KMM_INFO = check_accuracy(testY, pred_naive)
    # Naive Bayes Bernoulli
    classifier = ImportanceWeightedClassifier(iwe='kmm', loss="berno")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_KMM, acc_NB_KMM_INFO = check_accuracy(testY, pred_naive)
    #
    return pd.DataFrame(
            [{ 
            'window': window,
            'source_position': source_pos,
            'target_position': target_pos,

            'acc_LR_KMM': acc_LR_KMM,
            'acc_LR_KMM_INFO': str(acc_LR_KMM_INFO),                                              
            'acc_DT_KMM': acc_DT_KMM,
            'acc_DT_KMM_INFO': str(acc_DT_KMM_INFO),                                                                            
            'acc_NB_KMM': acc_NB_KMM,
            'acc_NB_KMM_INFO': str(acc_NB_KMM_INFO),                                                
            }]
        )

###
### Nearest-neighbour-based weighting (Loog, 2015)  
###
def apply_NN(trainX, trainY, testX, testY, window, source_pos, target_pos):
    # Decision Tree
    print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
    classifier = ImportanceWeightedClassifier(iwe='nn', loss="dtree")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_NN, acc_DT_NN_INFO = check_accuracy(testY, pred_naive)
    # Logistic Regression
    print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
    classifier = ImportanceWeightedClassifier(iwe='nn', loss="logistic")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_NN, acc_LR_NN_INFO = check_accuracy(testY, pred_naive)
    # Naive Bayes Bernoulli
    print("\n Nearest-neighbour-based weighting (Loog, 2015)    ")
    classifier = ImportanceWeightedClassifier(iwe='nn', loss="berno")
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_NN, acc_NB_NN_INFO = check_accuracy(testY, pred_naive)
    #
    return pd.DataFrame(
            [{ 
            'window': window,
            'source_position': source_pos,
            'target_position': target_pos,

            'acc_LR_NN': acc_LR_NN,
            'acc_LR_NN_INFO': str(acc_LR_NN_INFO),                                              
            'acc_DT_NN': acc_DT_NN,
            'acc_DT_NN_INFO': str(acc_DT_NN_INFO),                                                                            
            'acc_NB_NN': acc_NB_NN,
            'acc_NB_NN_INFO': str(acc_NB_NN_INFO),                                                
            }]
        )
#
# TCA
#
def apply_TCA(trainX, trainY, testX, testY, window, source_pos, target_pos):
    ####################################################
    ### Transfer Component Analysis (Pan et al, 2009) 
    ###
    # Decision Tree
    print("\n Transfer Component Analysis (Pan et al, 2009)")
    classifier = TransferComponentClassifier(loss="dtree", num_components=6)
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_DT_TCA, acc_DT_TCA_INFO = check_accuracy(testY, pred_naive)

    # Logistic Regression
    classifier = TransferComponentClassifier(loss="logistic", num_components=6)
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_LR_TCA, acc_LR_TCA_INFO = check_accuracy(testY, pred_naive)

    # Naive Bayes Bernoulli
    classifier = TransferComponentClassifier(loss="berno", num_components=6)
    classifier.fit(trainX, trainY, testX)
    pred_naive = classifier.predict(testX)
    acc_NB_TCA, acc_NB_TCA_INFO = check_accuracy(testY, pred_naive)

    return pd.DataFrame(
            [{ 
            'window': window,
            'source_position': source_pos,
            'target_position': target_pos,

            'acc_LR_TCA': acc_LR_TCA,
            'acc_LR_TCA_INFO': str(acc_LR_TCA_INFO),                                              
            'acc_DT_TCA': acc_DT_TCA,
            'acc_DT_TCA_INFO': str(acc_DT_TCA_INFO),                                                                            
            'acc_NB_TCA': acc_NB_TCA,
            'acc_NB_TCA_INFO': str(acc_NB_TCA_INFO),                                                
            }]
        )