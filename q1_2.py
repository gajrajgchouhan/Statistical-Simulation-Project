import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sns.set_theme()


class Estimate:
    """

    This class estimates using Monte Carlo Simulation in Question 1 and 2.
    It will also animate and save a GiF along with a confidence interval graph.

    Inputs:

    low, high -- Xlimit and Ylimit of the graph.
    l -- Number of points to take e.g. 1, 2, 10, 100, ....
    area_func -- Function to generate the points
    mask_func -- Function to mask the generate points to select the area in
                 which "useful" points will lie
    estimate_func -- Function to estimate the value from the masked points.

    E.g.

    Q1

    area_func - generated points in a 2x2 Square.
    mask_func - mask points where x^2 + y^2 <= 1 (radius of circle)
    estimate_func - pi = masked points / total points

    Q2

    area_func - generated points in 1x1 square.
    mask_func - mask points where for a point (x, y) f(x) is greater than equals y
    estimate_func - integral = masked points / total points

    """

    def __init__(self, low, high, l, area, mask, estimate, true_val):
        self.fig, self.ax = plt.subplots(1, 2, figsize=(6.4, 3.2))
        self.low = low
        self.high = high
        self.l = l  # points to take e.g. 100, 400 etc....
        self.time_steps = len(self.l)
        self.estimates = []  # estimates we make
        self.area_func = area  # generate the points in area
        self.mask_func = mask  # mask the area appropriately
        self.estimate_func = estimate  # func for estimating from mask
        self.true_val = true_val

    def init_func(self):
        for ax_ in self.ax.flatten():
            ax_.clear()

    def plotScatter(self, arr, ax, c="r"):
        ax.scatter(arr[:, 0], arr[:, 1], c=c)

    def __call__(self, i):
        self.init_func()

        points = self.l[i]
        estimates = []
        for _ in range(100):
            # generate 100 times to make sure we calculate the Confidence Interval
            area = self.area_func(self.low, self.high, points)
            mask = self.mask_func(area)
            masked = area[mask]
            estimate = self.estimate_func(self.high, self.low, masked, points)
            estimates.append(estimate)
        self.estimates.append(estimates)

        estimate = np.mean(estimates)

        self.ax[1].plot(self.l[: i + 1], np.mean(self.estimates, axis=1))
        self.ax[1].axhline(self.true_val, c="r")

        self.fig.suptitle(f"Points : {points} | Estimation = {estimate:.5}")
        self.plotScatter(area, self.ax[0])
        self.plotScatter(masked, self.ax[0], c="b")
        self.fig.tight_layout()

    def animate(self, name):
        ani = animation.FuncAnimation(
            fig=self.fig,
            func=self,
            init_func=self.init_func,
            interval=150,
            frames=self.time_steps,
        )
        ani.save(name, writer="ffmpeg")
        plt.close()

    def ci_graph(self, name, title=""):
        x = np.array(self.estimates)
        points = np.sqrt(self.l)
        mu = x.mean(axis=1)
        sd = np.sqrt(x.var(axis=1))
        ci = (1.645 * sd) / points  # calculate the confidence intervals
        plt.plot(self.l, mu)
        plt.title(title)
        plt.xlabel("Points")
        plt.ylabel("Estimate")
        plt.fill_between(self.l, (mu - ci), (mu + ci), alpha=0.5)
        plt.savefig(name)
        plt.close()


# Q1
# 3.14
low = -1.0
high = 1.0
l = list(range(1, 3001, 10))


def area(low, high, points):
    return np.random.default_rng().uniform(low, high, (points, 2))


def mask(area):
    return np.sum(area ** 2, axis=1) <= 1


def estimate(high, low, masked, points):
    return ((high - low) ** 2) * (masked.shape[0] / points)


sim = Estimate(low, high, l, area, mask, estimate, 3.14159)

sim.animate("1.mp4")
sim.ci_graph("1_ci.png", "Monte Carlo Estimates")

# --------------------------------------------------------------
# Q2
# 0.4815990593768372
low = 0.0
high = 1.0
l = list(range(1, 2001, 10))


def area(low, high, points):
    return np.random.default_rng().uniform(low, high, (points, 2))


def mask(area):
    f = (
        lambda x: (
            (-65536 * (x ** 8))
            + (262144 * (x ** 7))
            - (409600 * (x ** 6))
            + (311296 * (x ** 5))
            - (114688 * (x ** 4))
            + (16384 * (x ** 3))
        )
        / 27
    )
    val = f(area[:, 0])
    out = np.less_equal(val, area[:, 1])
    return (val >= 0) & ~out  # mask True where --- f(x) >= y and x >= 0


def estimate(high, low, masked, points):
    return ((high - low) ** 2) * (masked.shape[0] / points)


sim = Estimate(low, high, l, area, mask, estimate, 0.48159)
sim.animate("2.mp4")
sim.ci_graph("2_ci.png", "Monte Carlo Estimates")

# References

# https://stackoverflow.com/questions/63453435/animate-scatter-plot-with-colorbar-using-matplotlib
