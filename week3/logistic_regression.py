# -*- coding: utf-8 -*-
"""
This module implements the linear logistic algorithm and a visualized training demo.
In the algorithm, the feature x can be a vector, while the label y is continuous scalar.
It is highly recommended run the demo in a IDE(e.g. Spyder).
However, in order to run the demo, you can also type:
    $ python linear_regression.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

class LogisticRegression():
    """
    The Class simply implements the logistic regression algorithm.

    Note:
        Feature x can be mult-dim, with shape (dim, nums).
        Target y is a continuous scalar, with shape (nums, ).

    Attributes:
        dim (int): Dimension of feature.
        lr (float): Learning rate.
        batch_size (float): Mini-batch size.
        max_iter (int): Iteration number for stopping training.
        w (np.array, shape=(dim, )): Weight.
        b (float): Bias.
        plot_count(int): Number of plots, only used when the plots are saved.

    """
    def __init__(self, dim=2, lr=0.00018, batch_size=2500, max_iter=70):

        self.dim = dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

        self.w = np.zeros((dim, )) # can use different initialization methods
        self.b = 0

        self.plot_count = 0

    def infer(self, x_batch):

        return sigmoid(np.matmul(self.w, x_batch) + self.b)

    def cal_gradient(self, x_batch, gt_y_batch):

        pred_y_batch = self.infer(x_batch)

        dw = np.matmul(pred_y_batch - gt_y_batch, x_batch.T) / self.batch_size
        db = np.sum(pred_y_batch - gt_y_batch) / self.batch_size

        return dw, db

    def eval_loss(self, x_batch, gt_y_batch):

        pred_y_batch = self.infer(x_batch)
        return -np.sum(gt_y_batch*np.log(pred_y_batch)
                       + (1-gt_y_batch)*np.log(1-pred_y_batch)) / self.batch_size

    def dynamic_plot(self, x_total, gt_y_total, save_plot=False):
        
        # this function can only used for the dataset in heights_weights.cvs!
        
        dim, _ = x_total.shape
        assert dim == 2, "the dim of a single feature must be 2 in order to plot!"
        
        plt.figure(figsize=(8, 6), dpi=80) # generate canvas
        plt.ion() # open interactive mode
        plt.cla() # clear orignal picture
        plt.title("The Fitting Process of Logistic Regression") # set title
        plt.grid(True) # open grid line
        plt.xlabel("Height (inches)") # set x label name
        plt.ylabel("Weight (pounds)") # set y label name

        # plot
        #colors = get_color_label(gt_y_total)
        x_total = x_total.T
        plt.scatter(x_total[:, 0][:5000], x_total[:, 1][:5000], 
                    color="blue", label="Male")
        plt.scatter(x_total[:, 0][5000:], x_total[:, 1][5000:], 
                    color="green", label="Female")

        x0 = np.linspace(50, 85, num=120)
        x1 = - (self.w[0] * x0 + self.b) / self.w[1]

        plt.plot(x0, x1, 'r', label="Decision Boundary")
        plt.legend(loc="upper left", shadow=True) # set legend

        if save_plot:
            plt.savefig(str(self.plot_count) + ".png")
            self.plot_count += 1

        plt.pause(0.1)

    def train(self, x_total, gt_y_total, dynamic_plot=True, save_plot=False):

        dim, num_total = x_total.shape
        assert dim == self.dim, "the dim of x_total is not %d!\n" % self.dim

        for i in range(self.max_iter):

            if dynamic_plot:
                self.dynamic_plot(x_total, gt_y_total, save_plot=save_plot)

            batch_idxs = np.random.choice(num_total, self.batch_size)
            x_batch = np.stack([x_total[:, j] for j in batch_idxs], axis=-1)
            gt_y_batch = np.stack([gt_y_total[j] for j in batch_idxs])

            dw, db = self.cal_gradient(x_batch, gt_y_batch)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            print('current weight: {0}\ncurrent bias: {1}'.format(self.w, self.b))
            print('current loss: {}\n'.format(self.eval_loss(x_batch, gt_y_batch)))

        plt.ioff()

def preprocess(array):
    array[array == "Male"] = 1
    array[array == "Female"] = 0
    return array

def get_color_label(gt_y_total):

    return ['blue' if gt_y == 1 else 'green' for gt_y in gt_y_total]

def demo():
    """
    A visualized training process demo of logistic regression.
    It is highly recommended run the demo in a IDE(e.g. Spyder).
    """
    df = pd.read_csv("./heights_weights.csv")

    x_total = df.iloc[:, 1:3].values
    gt_y_total = preprocess(df.iloc[:, 0].values.flatten())

    solver = LogisticRegression(dim=2, lr=0.00018, batch_size=2500, max_iter=65)
    solver.train(x_total.T, gt_y_total, save_plot=True)

if __name__ == "__main__":
    demo()
