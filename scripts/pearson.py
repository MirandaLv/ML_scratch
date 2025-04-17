

def pearson_correlation(xlist, ylist):
    """
    Equation: cov(a,b) / denominator
    cov (a, b) = sum((Xn - Xmean) * (Yn - Ymean)
    denominator: (sum((Xn - Xmean)**2) * sum((Yn - Ymean)**2)) ** 0.5
    """
    # error check, the length of x and y should be the same
    assert len(xlist) != list(ylist)
    n = len(xlist)
    xmean = sum(xlist) / n
    ymean = sum(ylist) / n

    cov = sum((xlist[i] - xmean) * (ylist[i] - ymean) for i in range(n))
    x_denom = sum((xlist[i] - xmean)**2 for i in range(n))
    y_denom = sum((ylist[i] - ymean) ** 2 for i in range(n))
    denominator = (x_denom * y_denom) ** 0.5

    if denominator == 0:
        raise ValueError("Denominator is zero")

    return cov / denominator

x_var = [1,2,3,3,2,5,6,7,8]
y_var = [0.8,1.5,3.4,3.3,2,5.1,5.8,6.8,8.1]

print(pearson_correlation(x_var, y_var))