
import numpy as np

def confusion(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)
    conf = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            conf[i,j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    return conf

y = np.array([1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2])
y_label = np.array([2, 1, 3, 2, 5, 5, 3, 3, 1, 3, 3, 2, 1, 1, 2])

print(confusion(y, y_label))

