import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.sparse.construct import random
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# make_blobs(), a convenience function in scikit-learn used to 
# generate synthetic clusters

# "n_samples" is the total number of samples to generate.
# "centers" is the number of centers to generate.
# "cluster_std" is the standard deviation.
# "make_blobs()" returns a tuple of two values: 
# 1. A two-dimensional NumPy array with the x- and 
# y-values for each of the samples.
# 2. A one-dimensional NumPy array containing the 
# cluster labels for each sample
features, true_labels = make_blobs(
    n_samples = 200,
    centers = 3,
    cluster_std = 2.75,
    random_state = 42
)
# The "random_state" parameter is set to an integer value so
# you can follow the data presented in the tutorial. In practice, 
# it’s best to leave "random_state" as the default value,"None".

# Here’s a look at the first five elements for each of the variables 
# returned by "make_blobs()":
# print(features[:5])
print(true_labels[:5])
# A machine learning algorithm would consider weight more important 
# than height only because the values for weight are larger and have 
# higher variability from person to person.

# Machine learning algorithms need to consider all features on an 
# even playing field. That means the values for all features must 
# be transformed to the same scale. 
# The process of transforming numerical features to use the same 
# scale is known as feature scaling
# It’s an important data preprocessing step for most distance-based 
# machine learning algorithms because it can have a significant impact 
# on the performance of your algorithm.
# There are several approaches to implementing feature scaling.  
# In this example, you’ll use the "StandardScaler" class. 
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# The KMeans estimator class in scikit-learn is where you set the algorithm 
# parameters before fitting the estimator to the data. The scikit-learn 
# implementation is flexible, providing several parameters that can be tuned.
# "init" controls the initialization technique. The standard version of the 
# k-means algorithm is implemented by setting init to "random". Setting this 
# to "k-means++" employs an advanced trick to speed up convergence, which 
# you’ll use later.
# "n_clusters" sets k for the clustering step. This is the most important 
# parameter for k-means.
# "n_init" sets the number of initializations to perform. This is important 
# because two runs can converge on different cluster assignments. The default 
# behavior for the scikit-learn algorithm is to perform ten k-means runs and 
# return the results of the one with the lowest SSE.
# "max_iter" sets the number of maximum iterations for each initialization of
# the k-means algorithm.
# Instantiate the "KMeans" class with the following arguments:
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)
# Now that the k-means class is ready, the next step is to fit it to the data in 
# scaled_features. This will perform ten runs of the k-means algorithm on your 
# data with a maximum of 300 iterations per run:
kmeans.fit(scaled_features)
# Statistics from the initialization run with the lowest SSE are available as 
# attributes of kmeans after calling .fit():
# The lowest SSE value
print(kmeans.inertia_)
# Final locations of the centroid
print(kmeans.cluster_centers_)
# The number of iterations required to converge
print(kmeans.n_iter_)
# Finally, the cluster assignments are stored as a one-dimensional NumPy array 
# in kmeans.labels_. Here’s a look at the first five predicted labels:
print(kmeans.labels_[:5])
#  the ordering of cluster labels is dependent on the initialization. 
# Cluster 0 from the first run could be labeled cluster 1 in the 
# second run and vice versa. This doesn’t affect clustering evaluation 
# metrics.