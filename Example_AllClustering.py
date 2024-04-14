import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np

# Daten generieren
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.figure(figsize=(15, 10))

# # Hierarchisches Clustering mit Dendrogram
plt.subplot(3, 2, 1)
Z = linkage(X, 'ward')
dendrogram(Z)
plt.title('Hierarchisches Clustering')
plt.xlabel('x1')
plt.ylabel('x2')

# Partitionierendes Clustering mit KMeans
plt.subplot(3, 2, 2)
kmeans = KMeans(n_clusters=4, random_state=0)
labels_kmeans = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.5, marker='x')
plt.title('Partitioning-based Clustering (KMeans)')
plt.xlabel('x1')
plt.ylabel('x2')

# Density-based Clustering mit DBSCAN
plt.subplot(3, 2, 3)
dbscan = DBSCAN(eps=0.6, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', marker='o')
plt.title('Density-based Clustering (DBSCAN)')
plt.xlabel('x1')
plt.ylabel('x2')

# Grid-based Clustering (konzeptionelles Beispiel)
plt.subplot(3, 2, 4)
grid_size = 1.5
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_grid = np.arange(x_min, x_max, grid_size)
y_grid = np.arange(y_min, y_max, grid_size)
for x_line in x_grid:
    plt.axvline(x_line, color='gray', linestyle='--', linewidth=1)
for y_line in y_grid:
    plt.axhline(y_line, color='gray', linestyle='--', linewidth=1)
colors = np.array([(x - x_min) // grid_size + (y - y_min) // grid_size * len(x_grid) for x, y in X])
plt.scatter(X[:, 0], X[:, 1], c=colors, cmap='viridis', marker='o')
plt.title('Grid-based Clustering (konzeptionell)')
plt.xlabel('x1')
plt.ylabel('x2')



# Model-based Clustering mit GMM
plt.subplot(3, 2, 5)
gmm = GaussianMixture(n_components=4, random_state=0)
gmm.fit(X)
labels_gmm = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis', s=40, marker='o')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, alpha=0.5, marker='x')
plt.title('Model-based Clustering (GMM)')
plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.show()
