# -*- coding: utf-8 -*-
"""
This module implements the linear regression algorithm and a visualized training demo.
In the algorithm, a single data sample x can be a vector, while the label y is continuous scalar.
It is highly recommended run the demo in a IDE(e.g. Spyder).
You can also juts type following command to run the visualized demo:
    $ python linear_regression.py
"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

class LinearRegression():
    """
    The Class simply implements the linear regression algorithm.

    Note:
        Feature x can be mult-dim, with shape (dim, nums).
        Target y is a discrete scalar(value can only be 0 or 1), with shape (nums, ).

    Attributes:
        dim (int): Dimension of feature.
        lr (float): Learning rate.
        batch_size (float): Mini-batch size.
        max_iter (int): Iteration number for stopping training.
        w (np.array, shape=(dim, )): Weight.
        b (float): Bias.
        plot_count(int): Number of plots, only used when the plots are saved.

    """
    def __init__(self, dim=1, lr=0.00002, batch_size=50, max_iter=40):

        self.dim = dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

        self.w = np.zeros((dim, )) # can use different initialization methods
        self.b = 0

        self.plot_count = 0

    def infer(self, x_batch):

        return np.matmul(self.w, x_batch) + self.b

    def cal_gradient(self, x_batch, gt_y_batch):

        pred_y_batch = self.infer(x_batch)

        dw = np.matmul(pred_y_batch - gt_y_batch, x_batch.T) / self.batch_size
        db = np.sum(pred_y_batch - gt_y_batch) / self.batch_size

        return dw, db

    def eval_loss(self, x_batch, gt_y_batch):

        pred_y_batch = self.infer(x_batch)

        return 0.5 * np.sum((pred_y_batch - gt_y_batch)**2) / self.batch_size

    def dynamic_plot(self, x_total, gt_y_total, save=False):

        # this function can only plot 1-dim data
        dim, _ = x_total.shape
        assert dim == 1, "the dim of a single feature must be 1 in order to plot!"

        plt.figure(figsize=(8, 6), dpi=80) # generate canvas
        plt.ion() # open interactive mode
        plt.cla() # clear orignal picture
        plt.title("The Fitting Process of Linear Regression") # set title
        plt.grid(True) # open grid line
        plt.xlabel("x") # set x label name
        plt.ylabel("y") # set y label name

        # plot
        plt.scatter(x_total, gt_y_total, label='Ground Truth')
        plt.plot(x_total.flatten(), self.infer(x_total), 'r', label='Predict')
        plt.legend(loc="upper left", shadow=True) # set legend

        if save:
            plt.savefig(str(self.plot_count) + ".png")
            self.plot_count += 1

        plt.pause(0.3)

    def train(self, x_total, gt_y_total, dynamic_plot=True):

        dim, num_total = x_total.shape
        assert dim == self.dim, "the dim of x_total is not %d!\n" % self.dim

        for i in range(self.max_iter):
            if dynamic_plot:
                self.dynamic_plot(x_total, gt_y_total)

            batch_idxs = np.random.choice(num_total, self.batch_size)
            x_batch = np.stack([x_total[:, j] for j in batch_idxs], axis=-1)
            gt_y_batch = np.stack([gt_y_total[j] for j in batch_idxs])

            dw, db = self.cal_gradient(x_batch, gt_y_batch)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            print('current weight: {0}\ncurrent bias: {1}'.format(self.w, self.b))
            print('current loss: {}\n'.format(self.eval_loss(x_batch, gt_y_batch)))

        plt.ioff()

def gen_sample_data(dim=1, num_total=200):

    # give w and b certain value
    w = np.ones((dim, ))
    b = 10.5

    x_total = np.arange(num_total).reshape(1, num_total)
    gt_y_total = random.normal(np.matmul(w, x_total) + b, scale=10)
    return x_total, gt_y_total

def demo():
    """
    A visualized training process demo of linear regression.
    It is highly recommended run the demo in a IDE(e.g. Spyder).
    """
    x_total, gt_y_total = gen_sample_data(num_total=200) # generate training data

    solver = LinearRegression(dim=1, lr=0.00002, batch_size=50, max_iter=40)
    solver.train(x_total, gt_y_total)

if __name__ == "__main__":
    demo()
