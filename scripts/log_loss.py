

# https://medium.com/analytics-vidhya/understanding-the-loss-function-of-logistic-regression-ac1eec2838ce

import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    :param y_true: ground true label
    :param y_pred: a series of probability prediction
    :param eps: adjust value
    :return: loss value
    """
    # binary cross entropy, the loss function for a logistic regression
    # logloss = - 1/n (sum((Yi * logPi) + (1-Yi)*log(1-Pi)))
    assert len(y_true) == len(y_pred)

    n = len(y_true)
    loss = 0.0

    for i in range(n):
        # clip operation to prevent the prediction being 0 or 1 using a small eps value
        # ensure numerical stability when calculating the log loss
        p = min(max(y_pred[i], eps), 1 - eps)
        loss += y_true[i] * math.log(p) + (1 - y_true[i]) * math.log(1-p)

    return loss/n

y_true = [1, 0, 1, 1]
y_pred = [0.9, 0.1, 0.8, 0.7]

loss = log_loss(y_true, y_pred)
print(f"Log Loss: {loss}")

