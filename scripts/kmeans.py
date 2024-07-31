
"""
Kmeans is an unsupervised machine learning algorithm with the goals to cluster the input data into different groups based on the
similarity between each points.
- Input X, cluster = k,
- Randomly select k points from X as the initial cluster centroids.
- Calculating the distance of each point in X to the k centroids, and assign each point to the closest centroids.
- Recalculating the centroids based on the current points in each cluster.
- stops: 1. iteration number reached; 2. the centorids of each cluster do not change anymore.
- Export the label of each points in X.
"""


import numpy as np
from numpy.linalg import norm

def euclidean(x1, x2):
    return np.mean(np.sum((x1-x2)**2))

def cos_sim(x1, x2):
    return np.dot(x1,x2)/(norm(x1) * norm(x2))


class KMeans:

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters


    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialization and getting k number of points randomly selected from the current Xs (index)
        # dimension: random_sample_idx 1D
        random_sample_idx = np.random.choice(self.n_samples, self.k, replace=False) # the centoids cannot be selected more than once
        self.centroids = [self.X[idx] for idx in random_sample_idx] #

        # looping all the iteration
        # for each iteration, get the cluster index each x belongs to
        for i in range(self.max_iters):
            # assign x to the closest centroids
            self.clusters = self._create_clusters(self.centroids) # [[], [], []] include index of each point to the closet cluster 0,1,2

            # calculate new centroids from the current clusters
            old_centroids = self.centroids
            self.centroids = self._calc_new_centroid(self.clusters)

            # check if the current centroids are similar to the old centorids, if yes, then break the loop
            if self._check_centroids(old_centroids, self.centroids):
                break

        # the function will return the corresponding label each point belongs to
        return self._get_labels(self.clusters)

    def _get_labels(self, clusters):
        # This is a unsupervised classification model, so there is no y as input, the output label will be the corresponding
        # cluster's index number
        labels = np.empty(self.n_samples)

        for idx, cluster in enumerate(clusters):
            for x_idx in cluster:
                labels[x_idx] = idx

        return labels

    def _check_centroids(self, old, new):
        distances = [euclidean(old[i], new[i]) for i in range(self.k)]
        return np.sum(distances)==0 # if the centroid does not change anymore, break


    def _calc_new_centroid(self, clusters):

        # assign mean value of clusters to centroids
        centroids = np.zeros((self.k, self.n_features))

        for idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean

        return centroids


    def _create_clusters(self, centroids):
        # assign each point to the nearest cluster
        clusters = [[] for i in range(self.k)] # assign the index of each point to the nearest cluster

        for idx, i in enumerate(self.X):
            distances = [euclidean(center, i) for center in centroids]
            closest_idx = np.argmin(distances)
            clusters[closest_idx].append(idx)

        return clusters


if __name__== "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )

    clusters = len(np.unique(y))

    km = KMeans(k=3, max_iters=150)

    y_pred = km.predict(X)






