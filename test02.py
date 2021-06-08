# Choosing the Appropriate Number of Clusters
# you’ll look at two methods that are commonly 
# used to evaluate the appropriate number of 
# clusters:
# The elbow method
# The silhouette coefficient


import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.sparse.construct import random
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
features, true_labels = make_blobs(
    n_samples=200,
    centers= 3,
    cluster_std= 2.75,
    random_state= 42
)
print(true_labels[:5])

scaler = StandardScaler()
scaled_featurs = scaler.fit_transform(features)
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(scaled_featurs)
print(kmeans.inertia_)
print(kmeans.cluster_centers_)
print(kmeans.n_iter_)
print(kmeans.labels_[:5])
# To perform the elbow method, run several k-means, 
# increment k with each iteration, and record the 
# SSE:

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter":300,
    "random_state":42
}
# a list holds the sse values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_featurs)
    sse.append(kmeans.inertia_)

# There’s a sweet spot where the SSE curve starts to 
# bend known as the elbow point. The x-value of this 
# point is thought to be a reasonable trade-off between 
# error and number of clusters. In this example, the 
# elbow is located at x=3:
# plt.style.use("fivethirtyeight")
# plt.plot(range(1,11), sse)
# plt.xticks(range(1, 11))
# plt.ylabel("SSE")
# plt.xlabel("Number of clusters")
# plt.show()
# If you’re having trouble choosing the elbow point of 
# the curve, then you could use a Python package, kneed, 
# to identify the elbow point programmatically:
k1 = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)
print(k1.elbow)
# The silhouette coefficient is a measure of cluster cohesion 
# and separation.
# It quantifies how well a data point fits into its assigned 
# cluster based on two factors:
# How close the data point is to other points in the cluster
# How far away the data point is from points in other clusters
##############################################################

# Silhouette coefficient values range between -1 and 1. Larger 
# numbers indicate that samples are closer to their clusters 
# than they are to other clusters.  
# In the scikit-learn implementation of the silhouette coefficient,
# the average silhouette coefficient of all the samples is summarized 
# into one score. The "silhouette score()" function needs a minimum of 
# two clusters, or it will raise an exception. 
# Loop through values of k again. This time, instead of computing SSE, 
# compute the silhouette coefficient:
# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

for k in range(2,11):
    kmeans =KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_featurs)
    score = silhouette_score(scaled_featurs, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2,11))
plt.xlabel("Number of clusters")
plt.ylabel("silhouette coefficient")
plt.show()