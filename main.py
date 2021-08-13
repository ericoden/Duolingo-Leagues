import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# create stochastic
A = 0.7 * np.eye(10, k=0) + 0.2 * np.eye(10, k=1) + 0.1 * np.eye(10, k=-1)
A[0][0] = 0.8
A[9][9] = 0.9
print(A)
# get eigenvalues/eigenvectors
evals, evecs = np.linalg.eig(A)

max_eval = np.argmax(evals)
principle_evec = evecs[max_eval]

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

population = np.random.rand(10)

sns.set_style('whitegrid')

for i in range(100):
    population = np.matmul(population, A)
    sns.barplot(x=leagues, y=population)
    plt.xticks(rotation=45)
    plt.ylim([0, 10])
    plt.draw()
    plt.pause(0.01)
    plt.clf()
