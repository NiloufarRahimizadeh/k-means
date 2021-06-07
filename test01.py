import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
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
print(features[:5])
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