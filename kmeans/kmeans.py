import data_processing as dp
import data_processing as dp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import os

def compute_euclidean_distance(vec_1, vec_2,ax):
    return np.linalg.norm(vec_1 - vec_2, axis=ax)

# Create k centroids within random positions
def initialise_centroids(dataset, k):
    mean = np.mean(dataset, axis = 0)
    std = np.std(dataset, axis = 0)
    return (np.random.randn(k, dataset.shape[1]) * std + mean)


def kmeans(dataset, k):
    centroids = initialise_centroids(dataset, k)
    cluster_assigned = 0
    clusters = {centroid: [] for centroid in range(k)}

    for i in dataset:
        # Find closest cluster and sotre in dictionary
        cluster_assigned = get_closest_cluster(i, k, centroids)

        clusters[cluster_assigned].append(i)

    for i, c in enumerate(range(k)):
        centroids[c] = compute_centroids(clusters[c])

    return centroids, clusters


# Find the closest cluster to a data point
def get_closest_cluster(x, k, centroids):
    prev = 100000
    best = 0
    for i in range(k):
        cluster_distance = compute_euclidean_distance(x, centroids[i], 0)
        if cluster_distance < prev:
            prev = cluster_distance
            best = i
    return best


# recalculate values for centroids based on cluster assignments
def compute_centroids(data):
    return np.mean(data, axis=0)


def plot_data(data, centroids, clusters):
    plt.figure()
    _, ax = plt.subplots()

    colors=['b', 'g', 'y']
    for c in clusters:
        for j, x in enumerate(clusters[c]):
            ax.scatter(x[0], x[1], s=7, c=colors[c])
    
    ax.scatter(centroids[:,0], centroids[:,1], marker='*', s=150, c='r', label='centroid')
    plt.savefig(os.path.join('images', 'cluster.png'))


def main():
    df = dp.load_data('dog_breeds.csv')
    data = dp.data_norm(df)
    k = 3

    c_old = np.zeros((k, data[['height', 'tail length']].shape[1]))

    max_it = 10
    for i in range(max_it):
        print(f"Interation {i}")
        centroids, clusters = kmeans(data[['height', 'tail length']].values, k)

        error = compute_euclidean_distance(centroids, c_old, None)
        c_old = deepcopy(centroids)

        if error == 0:
            break


    plot_data(data, centroids, clusters)

if __name__ == '__main__':
    main()