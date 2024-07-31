


import numpy as np
from collections import Counter

def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=3):

        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x):

        labels = []
        for x_sample in x:
            distances = [euclidean(x_sample, x_train) for x_train in self.X_train]
            k_closest_idx = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[idx] for idx in k_closest_idx]
            pred_label = Counter(k_labels).most_common(1)[0][0]
            labels.append(pred_label)

        return labels


if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split


    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    c = 3
    clf = KNN(k=c)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(y_pred)




