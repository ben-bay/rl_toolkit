import numpy as np


def testbed(n_arms, mean, variance):
    action_values = np.random.normal(mean, variance, (1,n_arms))
    return action_values

print(testbed(10, 0, 1))
