# -*- coding: utf-8 -*-
"""
This module implements the kmeans++ algorithm and a visualized training demo.
It is highly recommended run the demo in a IDE(e.g. Spyder).
You can also juts type following command to run the visualized demo:
    $ python kmeans_plus_plus.py
"""

import numpy as np
from numpy import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KmeansPlusPlus():
    """
    The Class simply implements the kmeans++ algorithm.

    Parameters:
        n_clusters: int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate

        n_features: int, optional, default: 2
            The number of dataset X's features

        tol: float, optional, default: 1e-4
            Control early stopping based on the relative center changes with tol

        max_iter : int, optional
            Maximum number of iterations over the complete dataset before stopping
            independently of any early stopping criterion heuristics

    Attributes:
        cluster_centers_: array, [n_clusters, n_features]
            Coordinates of cluster centers

        plot_count: int
            The pictures that have plotted

    """
    def __init__(self, n_clusters=3, n_features=2, tol=1e-4, max_iter=300):

        self.n_clusters_ = n_clusters
        self.n_features = n_features
        self.tol_ = tol
        self.max_iter_ = max_iter

        self.cluster_centers_ = np.empty((n_clusters, n_features))
        self.plot_count = 0

    def cal_min_distances(self, X, cnt):

        distances = np.full((X.shape[0], ), np.inf)
        for i, x in enumerate(X):
            for j in range(cnt):
                distances[i] = min(distances[i],
                    np.sum(np.square(x - self.cluster_centers_[j])))

        return distances

    def init_cluster_centers(self, X, plot=True):

        # step 1) randomly choose a data point for the first cluster center
        self.cluster_centers_[0, :] = X[random.choice(range(X.shape[0]))]

        for cnt in range(1, self.n_clusters_):

            # step 2) Caltulate the min D(x)^2 to cluster centers for each data point
            distances = self.cal_min_distances(X, cnt)

            # step 3) Randomly choose a data point according to
            # the probability distribution of D(x)^2
            probs = distances / np.sum(distances)
            cumprobs = np.cumsum(probs)

            r = random.rand()
            for j, cumprob in enumerate(cumprobs):
                if r < cumprob:
                    i = j
                    break
            self.cluster_centers_[cnt, :] = X[i, :]
        if plot:
            print("Plot the initialized cluster centers:")
            self.plot(X, "Initialized")

    def fit(self, X, plot=True):
        """
        Parameters:
            X: array-like, shape = [n_samples, n_features]
                Coordinates of the data points to cluster
            plot: bool
                Decide whether to plot. Only support n_features of 2
        """
        n_samples, n_features = X.shape
        assert n_features == self.n_features, "the n_features of X is not %d!\n" \
            % self.n_features

        # step 1) Initializaion of cluster centers
        self.init_cluster_centers(X)
        for k in range(self.max_iter_):

            # step 2) Assignment of labels
            labels = [None] * n_samples
            for i, x in enumerate(X):
                min_distance = float("inf")
                for j, cluster_center in enumerate(self.cluster_centers_):
                    cur_distance = np.sum(np.square(x - cluster_center))
                    if cur_distance < min_distance:
                        min_distance = cur_distance
                        min_j = j

                labels[i] = int(min_j)

            # step 3) update cluster centers
            pre_cluster_centers = self.cluster_centers_
            self.cluster_centers_ = np.zeros((self.n_clusters_, n_features))
            nums = np.zeros((self.n_clusters_, ))
            for i, x in enumerate(X):
                self.cluster_centers_[labels[i], :] += x
                nums[labels[i]] += 1

            for j in range(self.n_clusters_):
                self.cluster_centers_[j, :] = self.cluster_centers_[j, :] / nums[j]

            if plot:
                print("Plot the cluster centers after iteration %d:" %(k+1))
                self.plot(X, "Iteration %d" %(k+1))

            if self.early_stop(pre_cluster_centers):
                break
        plt.ioff()

    def early_stop(self, pre_cluster_centers):

        for i in range(self.n_clusters_):
            if np.linalg.norm(self.cluster_centers_[i, :] - pre_cluster_centers[i, :]) \
                > np.linalg.norm(self.cluster_centers_[i, :]) * self.tol_:
                return False
        return True

    def plot(self, X, text="", save=True):
        assert self.n_features == 2, "the n_features must be 2 in order to visualized plot!"
        plt.figure(figsize=(8, 6), dpi=70) # generate canvas
        plt.ion() # open interactive mode
        plt.cla() # clear orignal picture
        plt.title("The Fitting Process of Kmeans++") # set title
        plt.grid(False) # close grid line
        plt.xlabel("x") # set x label name
        plt.ylabel("y") # set y label name

        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1],
                    marker='*', c="red", s=200, label='cluster centers')
        plt.legend(loc="upper left", shadow=True) # set legend
        plt.text(-1.5, -10, text, size=15, style="oblique", weight="light", alpha=0.5)

        if save:
            plt.savefig(str(self.plot_count) + ".png")
            self.plot_count += 1
        plt.pause(0.1)

def demo():
    """
    A visualized training process demo of kmeans++.
    It is highly recommended run the demo in a IDE(e.g. Spyder).
    """
    # Creating a sample dataset with 3 clusters
    # Set fixed random_state for repetition of same visualization
    X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=1)

    kmeans_plus_plus = KmeansPlusPlus(max_iter=20)
    kmeans_plus_plus.fit(X)

if __name__ == "__main__":
    demo()
