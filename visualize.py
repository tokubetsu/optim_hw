import numpy as np
from matplotlib import pyplot as plt


class OptimVisualizer:
    def __init__(self, func, rng, eps=0):
        self.func = func
        self.prepare_grid(rng, eps=eps)
        self.prepare_ax()

    def prepare_grid(self, xs, eps=0):

        mx1 = np.abs(xs[:, 0]).max() + np.max([np.abs(xs[:, 0]).std(), eps])
        mx2 = np.abs(xs[:, 1]).max() + np.max([np.abs(xs[:, 1]).std(), eps])

        x1_grid = np.linspace(-mx1, mx1, 100)
        x2_grid = np.linspace(-mx2, mx2, 100)

        self.x1, self.x2 = np.meshgrid(x1_grid, x2_grid)

        xs_new = np.hstack([self.x1.reshape(self.x1.size, -1), self.x2.reshape(self.x2.size, -1)])
        self.grid = self.func(xs_new.T).reshape(*self.x1.shape)

    def prepare_ax(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6.15))

        self.contours = self.ax.contour(self.x1, self.x2, self.grid, 30)
        self.ax.clabel(self.contours)

        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.set_title('Optim path')

    def draw_descent_path(self, xs, arr=False, show=True):
        lst = list(zip(*xs))

        self.ax.plot(*lst, label='descent path')
        self.ax.scatter(*lst, c='black', s=40, lw=0, label='descent steps')

        if arr:
            for i, xy in enumerate(zip(*lst)):
                if i == 0:
                    prev = xy
                else:
                    self.ax.annotate("", xy=xy, xytext=prev,
                                        arrowprops=dict(arrowstyle="->"))
                    prev = xy

        self.ax.legend()

        if show:
            plt.show()
        else:
            return self.ax
