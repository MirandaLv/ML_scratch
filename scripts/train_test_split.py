

import random
import numpy as np

def train_test_split(x, y, shuffle=True, split_rate=0.25, random_seed=None):

    n_array = len(x)
    # edge case
    if n_array == 0: raise ValueError("At least one array required as input")
    assert len(x) == len(y)
    combined = list(zip(x,y)) # combine the x, y pairs

    if random_seed is not None:
        random.seed(random_seed)

    if shuffle:
        random.shuffle(combined)

    x_shuffled, y_shuffled = zip(*combined)
    split_idx = int((1 - split_rate) * n_array)

    x_train = x_shuffled[:split_idx]
    x_test = x_shuffled[split_idx:]
    y_train = y_shuffled[:split_idx]
    y_test = y_shuffled[split_idx:]

    return x_train, x_test, y_train, y_test

X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

X_train, X_test, y_train, y_test = train_test_split(X, y, split_rate=0.4, random_seed=42, shuffle=False)

print("Train:", X_train, y_train)
print("Test:", X_test, y_test)


X = np.random.rand(30, 3) * 10  # values between 0 and 10
y = np.random.rand(30)

X_data = list(X)
y_data = list(y)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, split_rate=0.4, random_seed=42, shuffle=False)

print("Train:", X_train, y_train)
print("Test:", X_test, y_test)


def k_fold(x, y, k):
    """
    :param x: x features
    :param y: y predict variable
    :param k: k fold
    :return:
    """
    assert len(x) == len(y)
    assert k > 1 # at least 2 fold

    data = list(zip(x, y))
    n = len(data)
    fold_size = n // k # the size of each fold

    # shuffle data
    random.seed(42) # make sure reproducibility
    random.shuffle(data)
    # iterate each fold to be a validation dataset
    for i in range(k):
        start = i * fold_size # start index of the validation data
        end = start + fold_size if i != k - 1 else n
        valid_data = data[start:end]
        train_data = data[:start] + data[end:]

        X_train, y_train = zip(*train_data)
        X_val, y_val = zip(*valid_data)

        yield list(X_train), list(y_train), list(X_val), list(y_val)


# Example dataset
X = [[1], [2], [3], [4], [5], [1], [2], [3], [4], [5]]
y = [1, 0, 1, 0, 1, 1, 0, 1, 1, 1]

# 3-Fold CV
for i, (X_tr, y_tr, X_val, y_val) in enumerate(k_fold(X, y, 3)):
    print(f"Fold {i+1}:")
    print("Train:", X_tr, y_tr)
    print("Val:", X_val, y_val)
    print()