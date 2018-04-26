#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random

import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances


###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

featuresToUse = "surf"  # surf, CaffeNet4096, GoogleNet1024
numberIteration = 10
adaptationAlgoUsed = ["NA", "TCA"]
# see function adaptData for available algorithms

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################


def generateSubset(X, Y, nPerClass):
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        random.shuffle(idxClass)
        idx.extend(idxClass[0:min(nPerClass, len(idxClass))])
    return (X[idx, :], Y[idx])


def adaptData(algo, Sx, Sy, Tx, Ty):
    if algo == "NA":  # No Adaptation
        sourceAdapted = Sx
        targetAdapted = Tx
    elif algo == "SA":
        # Subspace Alignment, described in:
        # Unsupervised Visual Domain Adaptation Using Subspace Alignment, 2013,
        # Fernando et al.
        from sklearn.decomposition import PCA
        d = 80  # subspace dimension
        pcaS = PCA(d).fit(Sx)
        pcaT = PCA(d).fit(Tx)
        XS = np.transpose(pcaS.components_)[:, :d]  # source subspace matrix
        XT = np.transpose(pcaT.components_)[:, :d]  # target subspace matrix
        Xa = XS.dot(np.transpose(XS)).dot(XT)  # align source subspace
        sourceAdapted = Sx.dot(Xa)  # project source in aligned subspace
        targetAdapted = Tx.dot(XT)  # project target in target subspace
    elif algo == "OT":
        # Optimal Transport with class regularization described in:
        # Domain adaptation with regularized optimal transport, 2014.
        # Courty et al.
        import ot  # https://github.com/rflamary/POT
        transp3 = ot.da.SinkhornLpl1Transport(reg_e=2, reg_cl=1, norm="median")
        transp3.fit(Xs=Sx, ys=Sy, Xt=Tx)
        sourceAdapted = transp3.transform(Xs=Sx)
        targetAdapted = Tx
    elif algo == "TCA":
        # Domain adaptation via transfer component analysis. IEEE TNN 2011
        d = 80  # subspace dimension
        Ns = Sx.shape[0]
        Nt = Tx.shape[0]
        L_ss = (1. / (Ns * Ns)) * np.full((Ns, Ns), 1)
        L_st = (-1. / (Ns * Nt)) * np.full((Ns, Nt), 1)
        L_ts = (-1. / (Nt * Ns)) * np.full((Nt, Ns), 1)
        L_tt = (1. / (Nt * Nt)) * np.full((Nt, Nt), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        X = np.vstack((Sx, Tx))
        K = np.dot(X, X.T)  # linear kernel
        H = (np.identity(Ns+Nt)-1./(Ns+Nt)*np.ones((Ns + Nt, 1)) *
             np.ones((Ns + Nt, 1)).T)
        inv = np.linalg.pinv(np.identity(Ns + Nt) + K.dot(L).dot(K))
        D, W = np.linalg.eigh(inv.dot(K).dot(H).dot(K))
        W = W[:, np.argsort(-D)[:d]]  # eigenvectors of d highest eigenvalues
        sourceAdapted = np.dot(K[:Ns, :], W)  # project source
        targetAdapted = np.dot(K[Ns:, :], W)  # project target
    elif algo == "CORAL":
        # Return of Frustratingly Easy Domain Adaptation. AAAI 2016
        from scipy.linalg import sqrtm
        Cs = np.cov(Sx, rowvar=False) + np.eye(Sx.shape[1])
        Ct = np.cov(Tx, rowvar=False) + np.eye(Tx.shape[1])
        Ds = Sx.dot(np.linalg.inv(np.real(sqrtm(Cs))))  # whitening source
        Ds = Ds.dot(np.real(sqrtm(Ct)))  # re-coloring with target covariance
        sourceAdapted = Ds
        targetAdapted = Tx

    return sourceAdapted, targetAdapted


def getAccuracy(trainData, trainLabels, testData, testLabels):
    # ------------ Accuracy evaluation by performing a 1NearestNeighbor
    dist = euclidean_distances(trainData, testData, squared=True)
    minIDX = np.argmin(dist, axis=0)
    prediction = trainLabels[minIDX]
    accuracy = 100 * float(sum(prediction == testLabels)) / len(testData)
    return accuracy


# ---------------------------- DATA Loading Part ------------------------------
domainNames = ['amazon', 'caltech10', 'dslr', 'webcam']
tests = []
data = {}

for sourceDomain in domainNames:
    possible_data = loadmat(os.path.join(".", "features", featuresToUse,
                                         sourceDomain + '.mat'))
    if featuresToUse == "surf":
        # Normalize the surf histograms
        feat = (possible_data['fts'].astype(float) /
                np.tile(np.sum(possible_data['fts'], 1),
                        (np.shape(possible_data['fts'])[1], 1)).T)
    else:
        feat = possible_data['fts'].astype(float)

    # Z-score
    feat = preprocessing.scale(feat)

    labels = possible_data['labels'].ravel()
    data[sourceDomain] = [feat, labels]
    for targetDomain in domainNames:
        if sourceDomain != targetDomain:
            perClassSource = 20
            if sourceDomain == 'dslr':
                perClassSource = 8
            tests.append([sourceDomain, targetDomain, perClassSource])
meansAcc = {}
stdsAcc = {}
totalTime = {}
print("Feature used: ", featuresToUse)
print("Number of iterations: ", numberIteration)
print("Adaptation algorithms used: ", end="")
for name in adaptationAlgoUsed:
    meansAcc[name] = []
    stdsAcc[name] = []
    totalTime[name] = 0
    print(" ", name, end="")
print("")
# -------------------- Main testing loop --------------------------------------
for test in tests:
    Sname = test[0]
    Tname = test[1]
    perClassSource = test[2]
    print(Sname.upper()[:1] + '->' + Tname.upper()[:1], end=" ")
    # --------------------II. prepare data-------------------------------------
    Sx = data[Sname][0]
    Sy = data[Sname][1]
    Tx = data[Tname][0]
    Ty = data[Tname][1]
    # --------------------III. run experiments---------------------------------
    results = {}
    times = {}
    for name in adaptationAlgoUsed:
        results[name] = []
        times[name] = []
    for iteration in range(numberIteration):
        (subSx, subSy) = generateSubset(Sx, Sy, perClassSource)
        for name in adaptationAlgoUsed:
            # Apply domain adaptation algorithm
            startTime = time.time()
            subSa, Ta = adaptData(name, subSx, subSy, Tx, Ty)
            # Compute the accuracy classification
            results[name].append(getAccuracy(subSa, subSy, Ta, Ty))
            times[name].append(time.time() - startTime)
        print(".", end="")
    print("")
    for name in adaptationAlgoUsed:
        meanAcc = np.mean(results[name])
        stdAcc = np.std(results[name])
        meansAcc[name].append(meanAcc)
        stdsAcc[name].append(stdAcc)
        totalTime[name] += sum(times[name])
        print("     {:4.1f}".format(meanAcc) + "  {:3.1f}".format(stdAcc) +
              "  {:6}".format(name) + " {:6.2f}s".format(sum(times[name])))

print("")
print("Mean results and total time")
for name in adaptationAlgoUsed:
    meanMean = np.mean(meansAcc[name])
    meanStd = np.mean(stdsAcc[name])
    print("     {:4.1f}".format(meanMean) + "  {:3.1f}".format(meanStd) +
          "  {:6}".format(name) + " {:6.2f}s".format(totalTime[name]))
