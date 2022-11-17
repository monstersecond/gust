"""
Measure the distance between two spike sequence
"""
import torch
import numpy as np
import utils


def victor_purpura_metric(s1, s2, q=3):
    f1 = np.where(s1 > 0)[0]
    f2 = np.where(s2 > 0)[0]
    f1_len = len(f1)
    f2_len = len(f2)
    f = np.zeros((f1_len + 1, f2_len + 1))
    for i in range(f1_len + 1):
        f[i, 0] = i
    for j in range(f2_len + 1):
        f[0, j] = j
    for i in range(1, f1_len + 1):
        for j in range(1, f2_len + 1):
            f[i, j] = f[i-1, j-1] + abs(f1[i-1] - f2[j-1]) * q
            f[i, j] = min(f[i, j], f[i, j-1] + 1)
            f[i, j] = min(f[i, j], f[i-1, j] + 1)
    return f[f1_len, f2_len]
