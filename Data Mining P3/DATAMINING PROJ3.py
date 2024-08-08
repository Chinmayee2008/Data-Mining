#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os
import random
import math
import matplotlib.pyplot as plt


# Calculating the euclidean distance between points
def calculate_dist(point1, point2):
    total_dist = np.sum(pow(point2 - point1, 2))
    total_dist = math.sqrt(total_dist)
    return total_dist

# Function for calculating clusters
def k_means(k_cluster, data_pts):
    # Initializing a dictionary to store centroids
    centroid = {}
    for i in range(k_cluster):
        centroid[i] = random.choice(data_pts)
    # For loop for doing 20 iterations
    for _ in range(20):
        cluster = {}
        # Initializing an empty list to store records in each cluster
        for j in range(k_cluster):
            cluster[j] = []
        for record in data_pts:
            # Create a temp list to store the distances 
            distances = []
            # Calculate the distance 
            for m in range(k_cluster):
                distances.append(calculate_dist(centroid[m], record))
            # Finding the index of the minimum length in the distances list
            center = distances.index(min(distances))
            cluster[center].append(record)
        # Finding the new centroids based on clusters formed
        for p in range(k_cluster):
            centroid[p] = np.mean(cluster[p], axis=0)
    return cluster,centroid

file_input = input("Enter the filename:")
data_frame = pd.read_csv(file_input, delim_whitespace=True, header=None)
data_frame = data_frame.fillna(0)
data_pts = data_frame.iloc[:, :-1]
data_pts = data_pts.values
labels = data_frame.iloc[:, -1:]
k_error = []
for num_cluster in range(2, 11):
    clusters_result, centroids_result = k_means(num_cluster, data_pts)
    error = 0
    # Calculating the error after 20 iterations
    for i in range(num_cluster):
        for q in clusters_result[i]:
            error += calculate_dist(centroids_result[i], q)
    k_error.append(error)
    print("For K={}, After 20 iterations: SSE error = {:.4f}".format(num_cluster, error))

# Plotting the graph
x_values = list(range(2, 11))
plt.plot(x_values, k_error)
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("SSE vs K-Chart")
plt.show()

