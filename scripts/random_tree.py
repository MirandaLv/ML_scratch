
from decisionTree import DecisionTree
import numpy as np
from collections import Counter

"""
Decision tree is highly sensitive to the training data, which could results in high variance, the model might fail in generalization. 

Random forest is created with multiple number of trees, each trees are trained with randomly selected samples from the training dataset with replacement.
This process of creating a new training dataset is called bootstrapping. The trees are built with each of the resampled dataset.
Features for each tree are also randomly selected at each tree training step.  

Bootstrapping helps the model to be less sensitive to the training dataset.
Random feature selection helps to reduce the correlation between trees.
How many features are optimal for each tree? - log and sqrt of the total features.

"""
class RandomFoest:

    def __init__(self, n_trees=100, min_sample_splits=2, max_depth=100, n_feats=None):

        self.n_trees = n_trees
        self.min_sample_splits = min_sample_splits
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_sample_split=self.min_sample_splits, n_feats=self.n_feats)
            X_sample, y_sample = self.bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        all_predictions = np.moveaxis(all_predictions, 0, -1)
        forest_pred = [self._common_label(prediction) for prediction in all_predictions]
        return forest_pred


    def _common_label(self, y):
        counter = Counter(y)
        y_common = counter.most_common(1)[0][0]
        return y_common


    def bootstrap(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split


    def accuracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc


    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    clf = RandomFoest(n_trees=50, max_depth=50)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    pred_acc = accuracy(y_test, predictions)

    print("The random forest prediction accuracy is {}".format(pred_acc))

