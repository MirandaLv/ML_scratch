

"""
Maximize information gain at every split
Entropy: https://thisgoke.medium.com/entropy-a-method-for-data-science-machine-learning-7c3de2c6d82d
Measure the information contained in a state, if entropy is high, we are very uncertain about the randomly picked point
Entropy: sum(-Pi * log(Pi)) - Pi probability of class i
IG = E(parents) - sum(Wi * E(ChilDi)) (Wi here is the weight of each child node, influenced by the percentage of each class in the node
                    For example, if there are 20 total points, 4 are in one side, and 16 are in another side, the weights are
                    4/20 and 16/20, respectively.)

At each split, the model compares every possible split and take the one that maximize information gain

Greedy search

Gini index or Gini impurity
A metric that measures how mixed or impure a dataset is
Gini impurity for each node: G(t) = 1 - sum(Pi **2) [i=1-C]
            - Pi is the proportion of data points in node t belonging to class i, and c is the number of classes.
Weighted gini impurity for each split:
            - G(split) = (Nleft/N) * G(left) + (Nright/N) * G(right)


2. Decrease impurity or increase homogenity


"""



import numpy as np
from collections import Counter


class Node: # build a new node
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        self.value = value # the majority class of the leaf node

    # def is_leaf_node(self):
    #     # get the majority label of the leaf
    #     return



class DecisionTree:

    def __init__(self, max_depth=3, min_sample_split=2, n_feats=None):

        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_feats = n_feats # The number of features to consider when looking for the best split
        self.root = None


    def fit(self, X, y):

        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, curr_depth=0):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if curr_depth > self.max_depth or n_labels == 1 or n_samples <= self.min_sample_split:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_features, self.n_feats, replace=False)

        best_feat, best_threshold = self._best_criteria(X, y, feat_idx)

        # grow the children from the result split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_threshold)
        left = self.build_tree(X[left_idxs, :], y[left_idxs], curr_depth+1)
        right = self.build_tree(X[right_idxs, :], y[right_idxs], curr_depth + 1)

        return Node(best_feat, best_threshold, left, right)


    # get best split criteria
    def _best_criteria(self, X, y, feat_ids):

        best_gain = -1
        # looping the index of the feature selected
        for feat_id in feat_ids:
            X_column = X[:, feat_id] # filter the selected feature
            thresholds = np.unique(X_column) # get all the possible spliting threshold
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_id
                    split_threshold = threshold

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, split_thre, mode='entropy'):

        # child split
        left_idxs, right_idxs = self._split(X_column, split_thre)

        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0

        # compute the weights of each leaf
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        if mode == 'entropy':
            # parent information
            parent_e = self._entropy(y)
            e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
            child_e = (n_l/n) * e_l + (n_r/n) * e_r

            # calculate information gain
            ig = parent_e - child_e
        elif mode == 'gini':
            parent_g = self._gini(y)
            g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
            child_g = (n_l / n) * g_l + (n_r / n) * g_r

            # calculate information gain
            ig = parent_g - child_g

        return ig


    def _entropy(self, y):
        # Entropy = sum(-Pi * log(Pi))
        y_labels = np.unique(y)
        probs = [np.sum(y==cls)/len(y) for cls in y_labels]
        return -np.sum([p * np.log2(p) for p in probs if p > 0])


    def _gini(self, y):
        # Gini = 1 - sum(Pi **2)
        y_labels = np.unique(y)
        probs = [np.sum(y == cls) / len(y) for cls in y_labels]
        return 1 - np.sum([p**2 for p in probs if p > 0])


    def _split(self, X_column, split_thre):

        left_idxs = np.argwhere(X_column<split_thre).flatten()
        right_idxs = np.argwhere(X_column >= split_thre).flatten()

        return left_idxs, right_idxs

    def _leaf_value(self, y):
        counter = Counter(y)
        majority_label = counter.most_common(1)[0][0]
        return majority_label

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, node):
        if node.value != None: return node.value

        if x[node.feature] <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)

if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        acc = np.sum(y_true==y_pred) / len(y_true)
        return acc
        
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    pred_acc = accuracy(y_test, predictions)

    print(pred_acc)

    

















