

"""
1. Adaboost combines a lot of "weak learners" to make classification, the weak learners are always stumps.
2. Some stumps are getting more saying in the classification than other stumps.
3. Each stump is made by taking the previous stump's mistakes into account.

Stpes:
1. Give each sample in the training data a weight that indicates how important it is to be correctly classified. -> initial weight for each sample is equal
2. Make the first stump in the forest. For each feature in all features, find the feature and the threshold that best classify all samples (using gini).
3. Determine how much weight the first stump has in the final classification -> based on how well it classify the samples.
        The total error for a stump is the sum of weights associated with the incorrectly classified samples.
        Amount of say = (1/2) * log((1-Total error) / Total error)
4. Increasing the incorrectly classified samples weights, and decrease the correctly classified samples weights.
        New sample weight for incorrectly classified samples = old sample weight * e ** (amount of say)
        New sample weight for correctly classified samples = old sample weight * e ** (-amount of say)
        Normalize new sample weights, so they can add up to 1.
        Each sample weight divided by (New sample weight for incorrectly classified samples + New sample weight for correctly classified samples)
5. Randomly create a new training dataset, using the selection weights calculated above. The new training dataset has the same total
    amount of X train, but the samples with higher weights will be selected multiple times based on the weights.
    Now each sample in the new training set will have the same weight, and repeat the above steps.

    If we have a weighted gini function, then we use it with the sample weights.

References:
https://www.kaggle.com/code/vincentbrunner/ml-from-scratch-adaboost
https://www.youtube.com/watch?v=LsK-xG1cLYA

"""



import numpy as np


class DecisionStump:
    def __init__(self):
        # sample is classified either 1 or -1
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_sample = X.shape[0]
        X_column = X[:, self.feature_index]

        predictions = np.ones(n_sample)
        negative_idx = (self.polarity * X_column < self.polarity * self.threshold)
        predictions[negative_idx] = -1

        return predictions



class AdaBoost:

    def __init__(self, n_clf=5):
        """
        Adaboost combines a lot of "weak learners" to make classification, the weak learners are always stumps, which is a decision tree with max_depth = 1
        :param n_clf: The number of weak classifier
        """
        self.n_clf = n_clf

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initiate weights to each sample
        w = np.full(n_samples, (1/n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            # Minimize errors given thresholds to get the best threshold for features
            min_error = float('inf')

            # Iteration features and find the threshold
            for feat_id in range(n_features):
                thresholds = np.unique(X[:, feat_id])
                # Iterate each threshold
                for threshold in thresholds:
                    p = 1 # value to label correctly classified samples
                    # set all predictions to 1
                    predictions = np.ones(len(y))
                    # Label samples whose values are below threshold as -1
                    predictions[X[:,feat_id] < threshold] = -1
                    # calculating error: the error are the sum of the sample weights who are not correctly classified.
                    error = np.sum(w[y!=predictions])

                    # If the error is over 0.5, swap the label to -1 representing correctly classified.
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feat_id
                        min_error = error


            # Calculating the amount of saying for the stump
            # Adding a small correction parameter epsilon to prevent the denominator to be 0
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + EPS))

            # label samples that are correctly classified incorrectly classified
            predictions = np.ones(len(y))

            # if self.polarity == 1: predictions[predictions[clf.feature_index]<threshold] = -1
            # else: predictions[predictions[clf.feature_index]>threshold] = -1

            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1

            # update weights
            # incorrectly classified samples = old sample weight * e ** (amount of say)
            # correctly classified samples = old sample weight * e ** (- amount of say)
            w *= np.exp(- clf.alpha * predictions) # Check
            # normalize weight
            w /= np.sum(w)

            # save classifier
            self.clfs.append(clf)

    def predict(self, X):
        n_samples = X.shape[0]
        clf_predictions = [clf.alpha * clf.predict(X) for clf in self.clfs] # dont forget to multiply weights for each classifier
        y_pred = np.sum(clf_predictions, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy_score(y_true, y_pred):
        return np.sum(y_true==y_pred) / float(len(y_true))

    data = datasets.load_digits()
    X = data.data
    y = data.target

    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    # Change labels to {-1, 1}
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

    # Adaboost classification with 5 weak classifiers
    clf = AdaBoost(n_clf=7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # # Reduce dimensions to 2d using pca and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="Adaboost", accuracy=accuracy)


























