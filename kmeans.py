import random

def compute_euclidean_distance(vec_1, vec_2):
    # your code comes here
    distance = ((vec_1 ** 2) + (vec_2 ** 2))
    return distance


def initialise_centroids(dataset, k):
    # your code comes here
    centroids = []
    for i in range(k):
        print("set centroid")
    return centroids


def kmeans(dataset, k):
    # your code comes here
    centroids = initialise_centroids(dataset, k)
    cluster_assigned = 0
    for i in dataset:
        for j in range(k):
            compute_euclidean_distance(i, centroids[j])
    return centroids, cluster_assigned