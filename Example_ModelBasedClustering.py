import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generiere synthetische Daten
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=0)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

plt.figure(figsize=(10, 6))

cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

for i in range(4):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=cluster_names[i], cmap='viridis', s=40, marker='o')

plt.title('Model-based Clustering mit GMM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()

# Zentren der GMM-Komponenten
# plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, alpha=0.5, marker='x')