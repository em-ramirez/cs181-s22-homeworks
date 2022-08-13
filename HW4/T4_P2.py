# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as sc
import seaborn as sn

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
autograder_dataset = np.load("P2_Autograder_Data.npy")
large_dataset = np.load("data/large_dataset.npy")

np.random.seed(2)

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.clusters = {}
        self.distances = []
        self.erros = []

        for i in range(self.K):
            self.clusters[i] = []

    def euc_dist(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        self.X = X
        new_mu = np.random.rand(X.shape[1], self.K)
        old_mu = np.random.rand(X.shape[1], self.K)

        self.distances = [0] * self.K

        self.errors = []
        count = 0
        for _ in range(20):
            if count > 0:
                old_mu = np.array(new_mu).T

            for x in X:
                dist = [self.euc_dist(x, mu) for mu in old_mu.T]
                min_dist = np.argmin(dist)

                self.clusters[min_dist].append(x)
                self.distances[min_dist] = dist[min_dist]

            new_mu = []
            for i in range(self.K):
                new_mu.append(np.mean(self.clusters[i], axis=0))

            new_mu = np.array(new_mu)

            dists = 0
            for x in X:
                distances = [np.sum(np.square(x - mu)) for mu in new_mu]
                dists += min(distances)

            self.prev_assignments = np.argmin(dists)
            self.errors.append(dists)
            count += 1
        
    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        means = []
        for i in range(self.K):
            means.append(np.array(self.clusters[i]).mean(axis=0))

        return means
    
    def plot_errors(self):
        plt.plot(np.arange(len(self.errors)), self.errors)
        plt.show()

    def num_in_clusters(self, n_clusters):
        return [len(self.clusters[i]) for i in range(n_clusters)]

    def plot_num_in_cluster(self):
        num_in_clust = self.num_in_clusters(self.K)
        plt.bar(np.arange(len(num_in_clust)), num_in_clust)
        plt.xlabel("Cluster Index")
        plt.ylabel("Number of images in cluster")
        plt.title("K-Means Bar Chart")
        plt.legend()
        plt.show()

class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.prev_assignments = []
        self.distances = []
        self.clusters = None

    def linkage_dist(self, clust1, clust2):
        if self.linkage == 'min':
            return np.min(self.distances[clust1,:][:,clust2])
        elif self.linkage == 'max':
            return np.max(self.distances[clust1,:][:,clust2])
        else:
            return np.sqrt(np.sum((np.mean(self.X[clust1,:], 0) - np.mean(self.X[clust2,:], 0))**2))
    
    def merge_clusts(self, clust1, clust2):
        # reasssign labels
        small_label = np.min([clust1, clust2])
        large_label = np.max([clust1, clust2])
        self.clusters[self.clusters == large_label] = small_label
    
    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        self.X = X
        self.distances = sc.distance.cdist(X, X, 'euclidean')
        self.clusters = np.arange(self.X.shape[0])

        while len(np.unique(self.clusters)) > 1:
            opt_dist, clust1, clust2 = -1, -1, -1
            clusters = np.unique(self.clusters)

            count1 = 0
            for i in clusters:
                pts = (self.clusters == i).nonzero()[0]
                count2 = 0
                for j in clusters:
                    if count2 > count1:
                        pts2 = (self.clusters == j).nonzero()[0]
                        dist = self.linkage_dist(pts, pts2)
                        if opt_dist == -1 or dist < opt_dist:
                            opt_dist = dist
                            clust1, clust2 = i, j
                    count2 += 1
                count1 += 1
            
            self.merge_clusts(clust1, clust2)
            self.prev_assignments.append(np.copy(self.clusters))

    def n_clust(self, n_clusters):
        for i in self.prev_assignments:
            if len(np.unique(i)) == n_clusters:
                return i
        return None

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        n_clusts = self.n_clust(n_clusters)

        return (np.array([(self.X[np.array(n_clusts) == k]).mean(axis=0) for k in np.unique(n_clusts)]) if n_clusts is not None else [])
    
    def num_in_clusters(self, n_clusters):
        n_clusts = self.n_clust(n_clusters)
        
        return (np.unique(n_clusts, return_counts=True)[1] if n_clusts is not None else [])


# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
        KMeansClassifier.plot_errors()
        if i == 0:
            KMeansClassifier.plot_num_in_cluster()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.show()

# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)
# make_mean_image_plot(autograder_dataset, False)

# ~~ Part 3 ~~
# TODO: Change this line! standardize large_dataset and store the result in large_dataset_standardized
std = [1 if x == 0 else x for x in large_dataset.std(axis = 0)]
mean = large_dataset.mean(axis = 0)
large_dataset_standardized = (large_dataset - mean) / std

make_mean_image_plot(large_dataset_standardized, True)
# make_mean_image_plot(autograder_dataset, True)

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10
num_in_clust = []

fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    num_in_clust.append(hac.num_in_clusters(n_clusters))
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')

plt.show()


# TODO: Write plotting code for part 5
cluster_index = [i for i in range(n_clusters)]

plt.bar(cluster_index, num_in_clust[0])
plt.xlabel("Cluster Index")
plt.ylabel("Number of images in cluster")
plt.title("HAC bar chart for max linkage")
plt.show()

plt.bar(cluster_index, num_in_clust[1])
plt.xlabel("Cluster Index")
plt.ylabel("Number of images in cluster")
plt.title("HAC bar chart for min linkage")
plt.show()

plt.bar(cluster_index, num_in_clust[2])
plt.xlabel("Cluster Index")
plt.ylabel("Number of images in cluster")
plt.title("HAC bar chart for centroid linkage")
plt.show()


# TODO: Write plotting code for part 6
models = [(KMeans(10), "K-means"), (HAC('max'), "HAC max linkage"), (HAC('min'), "HAC min linkage"), (HAC('centroid'), "HAC centroid linkage")]
n_clusters = 10
confusion_mat = np.zeros((n_clusters, n_clusters))

clusts_per_model = []
for model in models:
    model[0].fit(small_dataset)
    if getattr(model[0], 'n_clust', False):
        clusts_per_model.append(model[0].n_clust(n_clusters))
    else:
        clusts_per_model.append(model[0].prev_assignments)
    
for c1 in range(len(models)):
    for c2 in range(c1 + 1, len(models)):
        for i in range(n_clusters):
            for j in range(n_clusters):
                confusion_mat[i, j] = np.sum((np.array(clusts_per_model[c1]) == i) * (np.array(clusts_per_model[c2]) == j))
        sn.heatmap(confusion_mat)
        plt.title("Heatmap for " + models[c1][1] + " and " + models[c2][1])
        plt.show()
