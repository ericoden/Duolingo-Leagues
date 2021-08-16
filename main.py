import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import time
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


# create stochastic
a = 13 / 30 * np.eye(10, k=0) +\
    (7 / 30) * np.eye(10, k=1) + \
    (10 / 30) * np.eye(10, k=-1)

a[0][0] = 15 / 30
a[1][0] = 15 / 30
a[0][1] = 7 / 30
a[1][1] = 8 / 30
a[2][1] = 15 / 30

a[9][9] = 23 / 30
print(np.around(a, 2))
# get eigenvalues/eigenvectors
w, v = np.linalg.eig(a)

max_eval = np.argmax(w)
principle_evec = w[max_eval]

# get random initial population
leagues = ['bronze',
           'silver',
           'gold',
           'sapphire',
           'ruby',
           'emerald',
           'amethyst',
           'pearl',
           'obsidian',
           'diamond']

colors = ["#cd7f32",
          "#c0c0c0",
          "#ffd700",
          "#0f52ba",
          "#e0115f",
          "#50c878",
          "#9966cc",
          "#eae0c8",
          "#000000",
          "#e0ffff"]

sns.set_palette(sns.color_palette(colors))

x0 = np.random.rand(10)

x0 = np.array([60, 90, 90, 0, 0, 0, 0, 0, 0, 0])

sns.set_style("darkgrid")

writer = PillowWriter(fps=25)

fig, ax = plt.subplots()
fig.set_tight_layout(True)
plt.xticks(rotation = 45)
bar = ax.bar(x=leagues, height=x0, color=colors)


def init():
    ax.set_xlabel("League", fontsize=20)
    ax.set_ylabel("Population", fontsize=20)
    ax.set_ylim(0, 100)
    return ax


def animate(i):
    m = np.linalg.matrix_power(a, i)
    # m = np.matmul(np.matmul(np.linalg.inv(v), np.diag(np.power(w, i))), v)
    x_i = np.matmul(m, x0)
    for j, b in enumerate(bar):
        b.set_height(x_i[j])


ani = FuncAnimation(fig,
                    animate,
                    frames=np.arange(0, 100),
                    repeat=True,
                    init_func=init)

ani.save('population_evolution.gif', writer=writer)
