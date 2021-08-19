import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sns.set_theme()

# Q6
# 2D Random Walk

x = [0]
y = [0]
fig, ax = plt.subplots(1, 1)


def gen_noise():
    # np.random.normal(0, 0.1)
    return np.random.choice([1, -1], size=2)


def animate(i):
    _x, _y = gen_noise()
    _x += x[-1]
    _y += y[-1]
    x.append(_x)
    y.append(_y)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.plot(x, y)
    ax.scatter(_x, _y, c="red")
    ax.scatter([0], [0], c="yellow")
    ax.set_title(f"Walk No: {i+1} | Mean = ({np.mean(x):.3}, {np.mean(y):.3})")


ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    init_func=(lambda: ax.clear()),
    interval=100,
    frames=500,
)
ani.save("6.mp4", writer="ffmpeg")
plt.close()
