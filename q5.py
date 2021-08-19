import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 125
mpl.rcParams["text.usetex"] = True
mpl.rc("font", **{"family": "sans-serif"})
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)

sns.set_theme()

# Q5
# Inverse Transform Sampling

pdf = np.vectorize(lambda x: (2 * x + 3) / 40)
inv_cdf = np.vectorize(lambda u: (40 * u + 9 / 4) ** 0.5 - 3 / 2)

for points in [1000]:
    u = np.random.default_rng().uniform(0.0, 1.0, (points, 1))
    samples = inv_cdf(u)
    plt.hist(samples, density=True, bins=20)
    plt.plot(samples, pdf(samples))
    plt.title(f"f(x) = $\\frac{{2x + 3}}{{40}}$ Points = {points}")
    plt.show()
