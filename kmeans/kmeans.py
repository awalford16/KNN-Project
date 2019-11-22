import data_processing as dp
import data_processing as dp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import os

def compute_euclidean_distance(vec_1, vec_2, ax):
    return np.linalg.norm(vec_1 - vec_2, axis=ax)

# Create k centroids within random positions
def initialise_centroids(dataset, k):
    mean = np.mean(dataset, axis = 0)
    std = np.std(dataset, axis = 0)
    return (np.random.randn(k, dataset.shape[1]) * std + mean)


def kmeans(dataset, k):
    centroids = initialise_centroids(dataset, k)
    cluster_assigned = 0
    #clusters = {centroid: [] for centroid in range(k)}

    max_it = 100
    errors = []
    c_old = np.zeros((k, dataset.shape[1]))
    #prev_error = 100
    for i in range(max_it):
        clusters = {centroid: [] for centroid in range(k)}
        c_old = deepcopy(centroids)

        print(f"Interation {i}")
        for i in dataset:
            # Find closest cluster and store in dictionary
            cluster_assigned = get_closest_cluster(i, k, centroids)
            clusters[cluster_assigned].append(i)

        for c in range(k):
            centroids[c] = compute_centroids(clusters[c])
            print(centroids[c])
        
        error = get_centroid_error(centroids, c_old)
        errors.append(error)

        if error == 0:
            break


    return centroids, clusters, errors


# Determine the error distance between new and old centroids
def get_centroid_error(c_new, c_old):
    return compute_euclidean_distance(c_new, c_old, None)


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


def plot_data(data, centroids, clusters, k):
    plt.figure()
    _, ax = plt.subplots()

    colors=['b', 'g', 'y']
    for c in clusters:
        for x in clusters[c]:
            ax.scatter(x[0], x[1], s=7, c=colors[c])
    
    ax.scatter(centroids[:,0], centroids[:,1], marker='*', s=150, c='r', label='centroid')
    plt.xlabel('height')
    plt.ylabel('tail length')
    plt.title(f"K-means Using Dog Height and Tail Length with K={k}")
    plt.savefig(os.path.join('images', 'cluster.png'))


def plot_error(error_data, k):
    plt.figure()
    plt.plot(error_data)
    plt.ylabel("error")
    plt.xlabel("iterations")
    plt.title(f"Iteration Error of Dog Height and Tail Length with K={k}")
    plt.savefig(os.path.join('images', 'kmean_error.png'))


def main():
    df = dp.load_data('dog_breeds.csv')
    data = dp.data_norm(df)
    k = 2

    centroids, clusters, errors = kmeans(data[['height', 'leg length']].values, k)

    print (errors)
    plot_data(data, centroids, clusters, k)
    plot_error(errors, k)

if __name__ == '__main__':
    main()