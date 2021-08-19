from math import pi, exp
import scipy
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# Q4
"""

Importance Sampling

"""

l = [100, 500, 1000, 2000, 3000, 5000, 10000]
sigma = 2

p = np.vectorize(lambda x: 0.5 * exp(-np.abs(x)))  # target distribution
g = np.vectorize(
    lambda x: (1 / (sigma * np.sqrt(2 * pi))) * exp(-0.5 * ((x / sigma) ** 2))
)  # proposed distribution


def expec(func, arr):  # for expectation
    f = (func(arr) * p(arr)) / g(arr)
    return np.var(f), np.mean(f)


funcs = {
    1: np.vectorize(lambda x: x),
    2: np.vectorize(lambda x: x ** 2),
    5: np.vectorize(lambda x: x ** 5),
}

results = np.zeros((len(l) * len(funcs), 6), dtype=object)
for i, points in enumerate(l):
    z = np.random.default_rng().normal(
        0, sigma, size=(points, 1)
    )  # proposed is normal
    for j, (key, val) in enumerate(funcs.items()):
        var_estimate, truth = expec(val, z)
        true_val = scipy.stats.laplace.moment(
            key, 0, 1
        )  # expectation == moment
        results[(i * len(funcs)) + j] = [
            points,
            f"E[x^{key}]",
            true_val,
            truth,
            abs(truth - true_val),
            var_estimate
        ]

results = pd.DataFrame(
    results, columns=["Points", "Expectation", "Truth", "Result", "Error", "Variance"]
).infer_objects()
results = results.sort_values(
    ["Expectation", "Points"], ascending=[True, True]
)

results.groupby(["Expectation"]).apply(print)
# results.groupby(["Expectation"]).apply(lambda df : print(df.to_latex(index=False)))
