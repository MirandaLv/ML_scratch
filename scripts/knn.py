
import numpy as np
from collections import Counter

def euclidean(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        all_labels = [self._predict(x_sub) for x_sub in x]
        return all_labels

    def _predict(self, x):
        # get an input x, and calculate the distance between x to each data in the training set
        alldistances = [euclidean(x, x_train) for x_train in self.X_train]
        # return the index that would sort an array
        k_idx = np.argsort(alldistances)[: self.k]
        # get the labels for point x_sub
        x_sub_labels = [self.y_train[idx] for idx in k_idx]
        x_sub_y_pred = Counter(x_sub_labels).most_common(1)
        # return Counter(words).most_common(10)
        #     [('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631),
        #     ('you', 554),  ('a', 546), ('my', 514), ('hamlet', 471), ('in', 451)]
        return x_sub_y_pred[0][0]

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("KNN classification accuracy", accuracy_score(y_test, predictions))





