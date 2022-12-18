import numpy as np
import scipy.stats as stat

# defining the function for the auto correlation test
def z_auto_corr(r: list, i: int, m: int, alpha: float = 0.05):
    """return ture if the null hypothesis is not rejected
    and false if the null hypothesis is rejected
    Parameters
    ----------
    r: list
        random numbers
    i: int
        start number
    m: int
        jump value
    alpha: float
        level of significance (default is 0.05)
    
    Returns
    -------
    tuple
        a tuple of boolean and a string determining the acceptance of h0
    """
    N = len(r)
    M = (N - m - i) // m
    test_nums = np.array(r)[[*range(i, N, m)]]
    p_hat = (1 / (M + 1)) * sum(test_nums[:-1] * test_nums[1:]) - 0.25
    sigma_hat = (13 * M + 7) ** 0.5 / (12 * (M + 1))
    z_0 = p_hat / sigma_hat
    z_c = stat.norm.ppf(1 - alpha / 2)
    return True, "hypothesis is not rejected" if z_0 > -z_c or z_0 < z_c else False, "hypothesis is rejected"

# list of random numbers to be tested for auto correlation
r = [0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
     0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
     0.68, 0.49, 0.05, 0.43, 0.95, 0.58, 0.19, 0.36, 0.69, 0.87]

# print result
print(z_auto_corr(r, 2, 5, 0.05))