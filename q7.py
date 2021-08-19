import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["text.usetex"] = True
mpl.rc("font", **{"family": "sans-serif"})
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)

sns.set_theme()

# Q7
# Multivariate Normal Distribution

points = 100000
mu = np.zeros(2)

mv_normal = (
    lambda cov: np.random.default_rng().multivariate_normal(mu, cov, points).T
)


def title(part, cov, mj=None, mn=None):
    t = (f"Part {part} | Covariance=$\\begin{{array}}{{cc}} {cov[0][0]}"
        f" & {cov[0][1]} \\\\ {cov[1][0]} & {cov[1][1]} \\end{{array}}$\n")

    if mj is not None and mn is not None:
        t += f" Minor Axis: {2*mn:.3},\nMajor Axis: {2*mj:.3}"
    return t


# Part A/B

fig, ax = plt.subplots(
    nrows=3, ncols=2, figsize=(10, 20), sharex=True, sharey=True
)

for c in range(1, 4):
    cov = c * np.identity(2)
    x, y = mv_normal(cov)
    mn, mj = max(abs(x)), max(abs(y))
    ax[c - 1, 0].scatter(x, y)
    ax[c - 1, 0].set_title(title("A", cov, mn=mn, mj=mj), x=1.5, y=0.5)

for c in range(1, 4):
    cov = c * np.diagflat([1, 2])
    x, y = mv_normal(cov)
    mn, mj = max(abs(x)), max(abs(y))
    s = ax[c - 1, 1].scatter(x, y)
    ax[c - 1, 1].set_title(title("B", cov, mn=mn, mj=mj), x=1.5, y=0.5)

for ax_ in ax.flatten():
    ax_.set_aspect("equal")
    ax_.set_xticks([])
    ax_.set_yticks([])

fig.tight_layout()
plt.show()
plt.close()

"""
As value of covariance matrix increases the radius of the circle/ellipse formed
    appears to be increasing.

In part A and part B, since the Σ matrix is a diagonal matrix.

The diagonal matrix's elements on main diagonal would specify the "variance"
    in the x and y direction.

We can see this is a perfectly symmetric curve in all the dimensions (also seen in the plots above).
"""

# Part C

cov = lambda a: np.array([[15, a], [a, 15]])

l = [1, 5, 8, 10, 14]

fig, ax = plt.subplots(
    nrows=len(l), ncols=2, figsize=(10, 5 * len(l)), sharex=True, sharey=True
)

for index, a in enumerate(l):
    mat = cov(a), cov(-a)
    x, y = mv_normal(mat[0])
    ax[index, 0].scatter(x, y)
    ax[index, 0].set_title(title("C", mat[0]), x=1.7, y=0.5)

    x, y = mv_normal(mat[1])
    ax[index, 1].scatter(x, y)
    ax[index, 1].set_title(title("C", mat[1]), x=1.7, y=0.5)

for ax_ in ax.flatten():
    ax_.set_aspect("equal")
    ax_.set_xticks([])
    ax_.set_yticks([])

fig.tight_layout()
plt.show()
plt.close()

"""
Here mean is still zero, although now covariance is a symmetric matrix.

We can see as magnitude of off diagonal elements increase the points
start to scatter in a rotated direction, and along that direction lies there maximum variance.

For negative elements, the direction is also opposite (as correlation will be negative).

This will also hold true for more dimensions.

This property of variance & direction of ellipse made by
multivariate normal distribution does not appear to be
for other distribution, although for a large number of
samples distributions will approximate to be like normal distribution
(due to CLT) hence it is applicable there.

"""
