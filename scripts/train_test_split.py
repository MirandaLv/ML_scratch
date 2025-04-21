

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

