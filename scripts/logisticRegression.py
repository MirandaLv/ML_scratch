

import numpy as np

class LogisticRegression:

    def __init__(self, max_iters=100, learning_rate=0.001):

        self.max_iters = max_iters
        self.lr = learning_rate

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iters):
            # calculate the y predict using current weights and bias, and update it with gradient decent
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid to make it logistic
            y_pred_cat = self._sigmoid(linear_model)

            # calculate gradient decent
            dw = (1/n_sample) * np.dot(X.T, (y_pred_cat - y))
            db = (1/n_sample) * np.sum(y_pred_cat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, x):

        linear_model = np.dot(x, self.weights) + self.bias
        y_preds = self._sigmoid(linear_model)
        y_pred_cls = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]
        return np.array(y_pred_cls)


    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def cal_precision(y_true, y_pred):
        return np.sum(y_true==y_pred)/len(y_true)

    bc = datasets.load_breast_cancer() # load the breast cancer dataset
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, max_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print(predictions)

    print(cal_precision(y_test, predictions))












