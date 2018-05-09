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
        #print("proposed_prediction:", proposed_prediction)
        for i in range(len(proposed_prediction[0])):
            voting = dict()
            #print("\nproposed_prediction[0]:", proposed_prediction[0])
            for col in range(len(proposed_prediction)):
                #print(i)
                label, prob = proposed_prediction[col][i]
                try:
                    values = voting[label][0] 
                    probs = voting[label][1]
                    voting[label] = (values+1, probs + prob) 
                except KeyError:
                    voting[label] = (1, prob)
                
                #final_predictions[i]
                #print(label, prob)

            #print("voting:", voting)
            max = (0,0)
            vote = ""
            for k in voting:
                #print("voting[k]:", k,  voting[k])
                if voting[k][0] > max[0]:
                    max = ( voting[k][0], ( voting[k][1]/voting[k][0] ) )
                    vote = k
                elif voting[k][0] == max[0] and ( ( voting[k][1]/voting[k][0] ) > max[1] ):
                    max = ( voting[k][0], ( voting[k][1]/voting[k][0] ) )
                    vote = k
                     
            #print(len(proposed_prediction[0]))  
            #print(i)             
            final_predictions[i] = vote
        #print("\nfinal_predictions:", final_predictions)
        return final_predictions

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
            probabilities[col] = alg.predict_proba(X)
            pred_prob = [0 for y in range(len(X))]
            for i, row in enumerate(pred_prob):
                pred_prob[i] = (predictions[col][i], np.max(probabilities[col][i]))
            final_predictions[col] = pred_prob
        #print(len(final_predictions), len(final_predictions[0]))
        #print(final_predictions)
        return self.majority_voting(final_predictions)