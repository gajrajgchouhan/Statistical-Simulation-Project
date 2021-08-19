from math import pi, exp
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mpl.rcParams["text.usetex"] = True
mpl.rc("font", **{"family": "sans-serif"})
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)

sns.set_theme()

# Q3
# Rejection Sampling

l = list(range(2, 2001, 10))
f = np.vectorize(
    lambda x: (1 / np.sqrt(2 * pi)) * exp(-0.5 * (x ** 2))
)  # target distribution
g = np.vectorize(
    lambda x: 1 / (pi * (1 + x ** 2))
)  # proposed distribution (standard cauchy)
M = 1.53

fig, ax = plt.subplots()


def animate(i):
    ax.clear()
    points = l[i]
    u = np.random.default_rng().uniform(0.0, 1.0, (points, 1)).reshape(-1, 1)
    x = np.array([])

    while len(x) < points:
        X = np.random.default_rng().standard_cauchy()
        if 0 <= abs(X) <= 10:
            x = np.append(x, X)
    x = x.reshape(-1, 1)
    r = (f(x) / g(x)) / M
    msk = np.ma.masked_where(u <= r, u)  # Points which will not be rejected

    plot_x = np.linspace(min(x), max(x), 500)

    ax.plot(
        plot_x,
        f(plot_x),
        linewidth=2,
        linestyle="-",
        color="green",
        label="f(x) = N(0, 1)",
    )

    ax.plot(
        plot_x,
        M * g(plot_x),
        linewidth=2,
        linestyle="-",
        color="red",
        label="g(x) = $\\dfrac{1}{\\pi * (1+x^2)}$",
    )

    ax.hist(x[msk.mask], bins=20, density=True)
    ax.set_xlim(-10, 10)

    ax.set_title(f"Points = {points}")
    ax.legend()


ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    init_func=(lambda: ax.clear()),
    interval=200,
    frames=len(l),
)

ani.save("3.mp4", writer="ffmpeg")
plt.close()
