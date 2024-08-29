

import numpy as np

# O(n)
def rotate_n(nums, k):
    """
    Do not return anything, modify nums in-place instead.
    """

    k = k % len(nums) # while k is larger than len(nums), the rotation number is k % len(nums)

    new_nums = list(np.zeros(len(nums)))

    for idx, val in enumerate(nums):
        new_idx = (idx + k) % len(nums)
        new_nums[new_idx] = val

    return new_nums

# O(1)

def flip_list(nums):
    l, r = 0, len(nums)-1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums



def rotate(nums, k):
    """
    Do not return anything, modify nums in-place instead.
    """
    k = k % len(nums) # while k is larger than len(nums), the rotation number is k % len(nums)

    l, r = 0, len(nums) - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1

    l, r = 0, k-1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1

    l, r = k, len(nums) - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1

    return nums

test = [1,2,3,4,5,6,7]
k = 3
print(rotate(test, k))





