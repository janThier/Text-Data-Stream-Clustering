import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generiere synthetische Daten
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Füge Ausreißer hinzu
X = np.vstack([X, np.array([[10, 10], [-10, -10], [10, -10], [-10, 10]])])  # Eindeutige Ausreißer

# Wende das Gaussian Mixture Model an
gmm = GaussianMixture(n_components=4, random_state=0)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

# Identifiziere Ausreißer
threshold = 0.1  # Schwellenwert für Ausreißer
outliers = np.max(probs, axis=1) < threshold

# Visualisiere die Datenpunkte
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, marker='o')

# Hervorheben der Ausreißer
plt.scatter(X[outliers, 0], X[outliers, 1], c='red', s=60, edgecolors='black', marker='o')

plt.title('Model-based Clustering mit GMM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Zeichne die Zentren der GMM-Komponenten
# plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, alpha=0.5, marker='x')