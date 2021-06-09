# Evaluating Clustering Performance Using Advanced Techniques

# The elbow method and silhouette coefficient evaluate 
# clustering performance without the use of ground truth labels.
# Ground truth labels categorize data points into groups based 
# on assignment by a human or an existing algorithm. 
# These types of metrics do their best to suggest the correct
# number of clusters but can be deceiving when used without context.
# When comparing k-means against a density-based approach on 
# nonspherical clusters, the results from the elbow method and 
# silhouette coefficient rarely match human intuition. This scenario 
# highlights why advanced clustering evaluation techniques are necessary. 
# To visualize an example, import these additional modules:
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# This time, use make_moons() to generate synthetic data in the shape 
# of crescents:

features, true_labels = make_moons(
    n_samples=250,
    noise=0.05,
    random_state=42
)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Fit both a k-means and a DBSCAN algorithm to the new data and visually 
# assess the performance by plotting the cluster assignments with 
# Matplotlib:
# DBSCAN: Density-based spatial clustering of applications with noise (DBSCAN) 
# is a data clustering algorithm
# Instantiate k-means and dbscan algorithms

kmeans = KMeans(n_clusters=2)
dbscan = DBSCAN(eps=0.3)

# Fit the algorithms to the features

kmeans.fit(scaled_features)
dbscan.fit(scaled_features)

# Compute the silhouette scores for each algorithm

kmeans_silhouette = silhouette_score(scaled_features, kmeans.labels_).round(2)

dbscan_silhouette = silhouette_score(scaled_features, dbscan.labels_).round(2) 

# Print the silhouette coefficient for each of the two algorithms and compare them. 
# A higher silhouette coefficient suggests better clusters, which is misleading in 
# this scenario:

# print(kmeans_silhouette)
# print(dbscan_silhouette)
# The silhouette coefficient is higher for the k-means algorithm. The DBSCAN algorithm 
# appears to find more natural clusters according to the shape of the data:
# Plot the data and cluster silhouette comparison

fig, (ax1, ax2) = plt.subplots( 
    1, 2, figsize=(8, 6), sharex=True, sharey=True
)  
fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
fte_colors = {
    0: "#008fd5",
    1: "#fc4f30",
}
# The k-means plot
km_colors = [fte_colors[label] for label in kmeans.labels_]
ax1.scatter(scaled_features[:,0], scaled_features[:,1], c=km_colors)
ax1.set_title(
    f"kmeans/nSilhouette:{kmeans_silhouette}", fontdict={"fontsize": 12}
)
# The dbscan plot
db_colors = [fte_colors[label] for label in dbscan.labels_]
ax2.scatter(scaled_features[:,0], scaled_features[:,1], c=db_colors)
ax2.set_title(
    f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
)
plt.show()
# This suggests that you need a better method to compare the performance of these two 
# clustering algorithms.
