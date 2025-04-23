

import numpy as np
import math

# Flatten a nested list of arbitrary depth
def flatten_arb_list(inlist):

    flat_list = []
    for lst in inlist:
        if isinstance(lst, list):
            flat_list.extend(flatten_arb_list(lst))
        else:
            flat_list.append(lst)

    return flat_list

nested = [1, [2, [3, 4], 5], [6, [7, [8]]]]
print(flatten_arb_list(nested))


# window function to average value within the windows
def rolling_avg(data, window_size):
    # edge case
    if window_size <= 0: raise ValueError("Window size must be positive")
    if window_size > len(data): raise ValueError("Window size cannot be larger than the data length")

    result = []
    window_sum = sum(data[:window_size])
    result.append(window_sum / window_size)

    for i in range(window_size, len(data)):
        window_sum = window_sum + data[i] - data[i-window_size]
        result.append(window_sum / window_size)

    return result

data = [1, 2, 3, 4, 5, 6, 7]
window_size = 3
print(rolling_avg(data, window_size))


def cosine_similarity(v1, v2):
    assert len(v1) == len(v2)
    # compute dot product
    dot_product = sum(a * b for a,b in zip(v1, v2))
    mag_a = sum(a * a for a in v1) ** 0.5
    mag_b = sum(b * b for b in v2) ** 0.5

    if mag_a == 0 or mag_b == 0: return 0.0

    return dot_product / (mag_a * mag_b) # return the cosine similarity

def cosine_sim_np(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0

    return dot_product / (norm_v1 * norm_v2)

v1 = [1, 2, 3]
v2 = [4, 5, 6]

print(cosine_similarity(v1, v2))
print(cosine_sim_np(v1, v2))


def calc_impurity(labels):
    n = len(labels)
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    impurity = 1.0
    # calculate the proportion of each label
    for count in label_counts.values():  # Gini = 1 - Σ(p_i)²
        prob = count/n
        impurity -= prob ** 2
    return impurity

def calc_impurity_np(labels):
    _, counts = np.unique(labels, return_counts=True)
    prob = counts / counts.sum()
    return 1 - np.sum(prob**2)


# Entrop (S) = -∑ (pi * log2(pi))
def calc_entropy(labels):
    n = len(labels)
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    entropy_val = 0.0
    for count in label_counts.values():
        prob = count / n
        if prob > 0:
            entropy_val -= prob * (math.log(prob) / math.log(2))
    return entropy_val

def calc_entropy_np(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return - np.sum(probs * np.log2(probs + 1e-9)) # add epsilon to avoid log(0)


def information_gain(parent, left, right):
    # compare the parent impurity, and the left/right impurity
    parent_entro= calc_entropy_np(parent)
    left_entro = calc_entropy_np(left)
    right_entro = calc_entropy_np(right)

    info_gain = parent_entro - (len(left)/len(parent) * left_entro + len(right)/len(parent) * right_entro)
    return info_gain


parent = [0, 0, 1, 1, 1, 1]
left = [0, 0]
right = [1, 1, 1, 0]

print("Gini (parent) without using numpy:", calc_impurity(parent))
print("Gini (parent) using numpy:", calc_impurity_np(parent))

print("Entropy (parent) without using numpy:", calc_entropy(parent))
print("Entropy (parent) using numpy:", calc_entropy_np(parent))

print("Information Gain:", information_gain(parent, left, right))

def cal_sigmoid(x): # σ(x)= 1 / (1 + e ** (-x))
    if isinstance(x, list):
        return [1 / (1 + math.exp(-xi)) for xi in x]
    else:
        return 1 / (1 + math.exp(-x))

def cal_softmax(x):
    max_x = max(x) # get the maximum value of x, each number minus the max value to ensure numerical stability
    exps = [math.exp(xi - max_x) for xi in x]
    sum_exps = sum(exps)
    return [ex / sum_exps for ex in exps]

# meaning that the model is about 50% confidence that the input belongs to the positive class (label=1)
print("Sigmoid(0):", cal_sigmoid(0))
print("Sigmoid([0, 1, -1]):", cal_sigmoid([0, 1, -1]))

print("Softmax([2.0, 1.0, 0.1]):", cal_softmax([2.0, 1.0, 0.1]))
# [0.6590011388859679, 0.24243297070471392, 0.09856589040931818]
# meaning that it is 65% confidence that the class belongs to 0
# 24% confidence belong to 1, and 9% confidence that it belongs to 2




