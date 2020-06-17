import os
import numpy as np
from sklearn import cluster, covariance, manifold
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import euclidean_distances


def cluster_AP(X, pref):
    af = AffinityPropagation(preference=pref).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    afmatrix = -af.affinity_matrix_
    centers = af.cluster_centers_
    n_clusters = len(cluster_centers_indices)
    print("Estimated number of clusters: %d" % n_clusters)
    # calculate distance from each cluster centriod
    dist = []
    for i in range(len(labels)):
        dist.append(afmatrix[cluster_centers_indices[labels[i]]][i])
    return list(zip(labels, dist)), n_clusters, afmatrix


def cluster_Kmeans(X, num):
    kmeans = KMeans(n_clusters=num)
    trans = kmeans.fit_transform(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    dist = []
    for i in range(len(labels)):
        dist.append(trans[i][labels[i]])
    return list(zip(labels, dist)), num, trans


def multiD_scale(centers, lables):
    # multi-dimensional scaling
    centers -= centers.mean()
    similarities = euclidean_distances(centers)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_
    # Rescale the data
    pos *= np.sqrt((centers ** 2).sum()) / np.sqrt((pos ** 2).sum())
    pos = [x +[int(y[0])] for x, y in zip(pos.tolist(), lables)]
    #print pos
    return pos
