
import numpy as np

def calc_median(nums):
    """Calculate the median value of a list"""
    if not nums: return None

    sorted_nums = sorted(nums)
    n = len(sorted_nums)

    if n % 2 == 0: # even number of list
        mid = n // 2
        prev = mid - 1
        median_val = (sorted_nums[mid] + sorted_nums[prev]) / 2
    else:
        mid = n // 2
        median_val = sorted_nums[mid]

    return median_val

def calc_mode(nums):
    """Calculate the mode value of a list"""

    if not nums: return None # edge case

    hashmap = {}
    for num in nums:
        hashmap[num] = hashmap.get(num, 0) + 1

    max_freq = max(hashmap.values())

    res = [] # it could be multiple result
    for k, v in hashmap.items():
        if v == max_freq:
            res.append(k)

    return res

def calc_mean(nums):
    if not nums: return None
    return np.array(nums).mean()

def calc_sd(nums):
    std = np.array(nums).std()
    return std

def calc_var(nums):
    var = np.array(nums).var()
    return var

example = [1,2,3,3,2,5,6,7,8]
example1 = []

print(calc_median(example))
print(calc_mode(example))
print(calc_mean(example))
print(calc_sd(example))
print(calc_var(example))
