
def min_max_normalization(data): # input is a list of data
    if not data: return None
    min_val = min(data)
    max_val = max(data)

    if min_val == max_val: # prevent the denominator to be 0
        return [0.0 for _ in data]

    return [(x - min_val) / (max_val - min_val) for x in data]


def zscore_normalization(data):
    if not data: return data
    mean_val = sum(data) / len(data)
    variance = sum((x - mean_val)**2 for x in data) / len(data)
    std_ = variance ** 0.5

    if std_ == 0:
        return [0.0 for _ in data]

    return [(x - mean_val) / std_ for x in data]

var = [0.8,1.5,3.4,3.3,2,5.1,5.8,6.8,8.1]

print(min_max_normalization(var))
print(zscore_normalization(var))