import os
import random
import statistics
from sklearn.cluster import KMeans, AgglomerativeClustering
import math
import matplotlib.pyplot as plt
import numpy.random
import pandas as pd
import datetime


def calculate_distance(x, y):
    """
    Calculates the euclidean distance between the two vectors
    :param x: The first vector
    :param y: The second vector
    :return: The calculated euclidean distance between x and y
    """
    euclidean = 0
    for i, j in zip(x, y):
        euclidean += (j - i) ** 2
    return math.sqrt(euclidean)


def choose_initial_centroids(cluster, k):
    """
    Chooses initial centroids based on the k-Means++ algorithm
    :param cluster: The cluster to choose centroids for
    :param k: The number of centroids to find
    :return: The list of centroids
    """
    centroid_list = [random.choice(cluster)[:-1]]
    for i in range(1, k + 1):
        distance_list = [[] for _ in range(len(cluster))]
        for j, item in enumerate(cluster):
            for centroid in centroid_list:
                distance_list[j].append(calculate_distance(item[:-1], centroid[:-1]))
        distance_list = [min(dist) for dist in distance_list]
        distance_list = [dist ** 2 for dist in distance_list]
        s = sum(distance_list)
        distance_list = [dist / s for dist in distance_list]
        centroid_list.append(cluster[numpy.random.choice(361, p=distance_list)][:-1])
    return centroid_list


def mean_of_cluster(cluster):
    """
    Finds the centroid of the given cluster
    :param cluster: The cluster to find the centroid of
    :return: The centroid of the cluster
    """
    result = []
    for i in range(3):
        result.append(statistics.mean(l[i] for l in cluster))
    return result


def within_cluster_sum_of_squares(cluster, centroid):
    """
    Calculates the WSS metric for the given cluster
    :param cluster: The cluster to calculate
    :param centroid: The centroid of the cluster
    :return: the WSS for the cluster
    """
    result = []
    for item in cluster:
        result.append(calculate_distance(item, centroid) ** 2)
    return sum(result)


def between_cluster_sum_of_squares(clusters, centroid_all):
    """
    Calculates the BSS metric for the given clusters
    :param clusters: The set of all clusters
    :param centroid_all: The centroid of all of the data
    :return: The BSS metric for the given clusters
    """
    result = []
    for cluster in clusters:
        result.append(len(cluster) * calculate_distance(centroid_all, mean_of_cluster(cluster)) ** 2)
    return sum(result)


def extract_features():
    data = pd.read_csv("Updated_Data.csv")
    data = data.loc[:, ["Moving Time", "Distance (km)", "Calories", "Type"]]
    for i in range(len(data["Moving Time"])):
        x = datetime.datetime.strptime(data["Moving Time"][i], '%H:%M:%S')
        data["Moving Time"][i] = datetime.timedelta(hours=x.hour,minutes=x.minute,seconds=x.second).total_seconds()
    return data


def k_means_clustering(feature_vectors, k):
    """
    Performs k-Means clustering for the given feature vectors with k clusters
    :param feature_vectors: The vectors to cluster
    :param k: The number of clusters to create
    :return: The list of clusters, list of corresponding centroids, the WCSS metric
    """
    clusters = [[] for _ in range(k)]
    centroid_list = choose_initial_centroids(feature_vectors, k)
    master_centroid_list = [centroid_list]
    # 10 is the maximum iterations for clustering
    for i in range(10):
        clusters = [[] for _ in range(k + 1)]
        for data_point in feature_vectors:
            compare_list = [0 for _ in range(k + 1)]
            for j, centroid in enumerate(centroid_list):
                compare_list[j] = calculate_distance(data_point[:-1], centroid)
            clusters[compare_list.index(min(compare_list))].append(data_point)
        centroid_list = [mean_of_cluster(cluster) for cluster in clusters]
        if centroid_list in master_centroid_list:
            break
        master_centroid_list.append(centroid_list)
    wcss = sum([within_cluster_sum_of_squares(cluster, centroid) for cluster, centroid in
                zip(clusters, centroid_list)])
    final_result = []
    for item in feature_vectors:
        for cluster in clusters:
            if item in cluster:
                final_result.append(clusters.index(cluster))
    return clusters, centroid_list, wcss


test = extract_features()

result_clusters, centroid_list, wcss = k_means_clustering(test.values.tolist(), 1)
for i in range(len(result_clusters)):
    for j in range(len(result_clusters[i])):
        result_clusters[i][j].append(i)
cluster1 = pd.DataFrame(result_clusters[0] + result_clusters[1], columns=["Moving Time (s)", "Distance (km)", "Calories", "Type", "Cluster"])
cluster1.to_csv("2_means_results.csv")
print("Clustering results written to 2_means_results.csv")