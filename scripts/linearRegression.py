
import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.001, max_iters=100):

        self.lr = learning_rate
        self.max_iters = max_iters


    def fit(self, X, y):

        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_sample) * np.dot(X.T, (y_pred - y))
            db = (1/n_sample) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, x):
        return np.dot(x, self.weights) + self.bias



if __name__=="__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    def mse(x1, x2):
        return np.mean((x1 - x2)**2)


    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.01, max_iters=1000)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print(mse(y_pred, y_test))

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()



