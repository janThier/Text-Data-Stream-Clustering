from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Datenpunkte generieren
np.random.seed(42)
X = 0.7 * np.random.randn(150, 2)

# outlier hinzufügen
X = np.vstack([X, np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]])])

# density-based Clustering
dbscan = DBSCAN(eps=0.3)
clusters = dbscan.fit_predict(X)

plt.figure(figsize=(10, 6))
unique_labels = set(clusters)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # rot für outlier
        col = [1, 0, 0, 1]

    class_member_mask = (clusters == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6, label='Cluster {}'.format(k) if k != -1 else 'Outliers')

plt.title('Density-based Clustering (DBSCAN)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend(loc='upper left')

plt.show()