from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Beispiel-Datensatz erstellen
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Clustering durchführen
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
y_pred = kmeans.predict(X)

# SC misst wie ähnlich ein Objekt zum eigenen Cluster ist (1 = perfekte Trennung)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, y_pred)) # 0.682

# DBI misst die Kompaktheit und Trennschärfe der Cluster (0 = perfekte Kompaktheit)
print("Davies-Bouldin Index: %0.3f"% metrics.davies_bouldin_score(X, y_pred)) # 0.438

# CHI ist ein Maß, das die Streuung betrachtet (je höher, desto besser die Trennschärfe)
print("Calinski-Harabasz Index: %0.3f" % metrics.calinski_harabasz_score(X, y_pred)) # 1210.090

# NMI misst die Übereinstimmung der wahren und der vorhergesagten Label (1 = perfekte Übereinstimmung)
print("Normalized Mutual Information: %0.3f"% metrics.normalized_mutual_info_score(y_true, y_pred)) # 1.000

# Homogeneity misst, ob die Cluster nur Datenpunkte aus einer einzigen Klasse enthalten
print("Homogeneity: %0.3f"% metrics.homogeneity_score(y_true, y_pred)) # 1.000

plt.figure(figsize=(10, 6))
cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

# jeden Datenpunkt separat zeichnen und entsprechenden Label-Parameter hinzufügen
for i in range(4):
    plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], label=cluster_names[i], s=50)

# Zentren der Cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids', marker='x')

plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()
