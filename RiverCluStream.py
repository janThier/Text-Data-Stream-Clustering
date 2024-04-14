from river import cluster
from river import stream

import matplotlib.pyplot as plt
import numpy as np

X = [
    [1, 2], [1, 4], [1, 0],
    [-4, 2], [-4, 4], [-4, 0],
    [5, 0], [5, 2], [5, 4],
    [2, 2], [2, 4], [2, 0],
    [-3, 2], [-3, 4], [-3, 0],
    [6, 0], [6, 2], [6, 4]
]

clustream = cluster.CluStream(
    n_macro_clusters=3, # Anzahl der Macro-Cluster
    max_micro_clusters=5, # maximale Anzahl der Micro-Cluster
    time_gap=3, # Zeitabstand zwischen macro-clustering durch inkrementelles k-means
    seed=0, # Seed für die Zufallszahlengenerierung der initialen Clusterzentren
    halflife=0.4 # Menge, um die die Clusterzentren verschoben werden (zw. 0 und 1)
)

cluster_ids = []

for x, _ in stream.iter_array(X):
    # Die Methode learn_one() fügt ein feature-set zum CluStream-Modell hinzu
    clustream.learn_one(x)
    cluster_id = clustream.predict_one({0: x[0], 1: x[1]})
    cluster_ids.append(cluster_id)

print(clustream.predict_one({0: 1, 1: 1})) # 0
print(clustream.predict_one({0: -4, 1: 3})) # 2
print(clustream.predict_one({0: 4, 1: 3.5})) # 0

# # Extrahiere die Zentren der Macro-Cluster
centers = clustream.centers

# # Erstelle eine Liste für x- und y-Koordinaten der Cluster-Zentren
# centers_x = [center[0] for center in centers.values()]
# centers_y = [center[1] for center in centers.values()]

# Erstelle eine Liste für x- und y-Koordinaten deiner Datenpunkte
data_x = [point[0] for point in X]
data_y = [point[1] for point in X]

# Erstelle den Scatter-Plot
plt.figure(figsize=(10, 6))

# Erstelle eine Farbpalette
colors = plt.cm.viridis(np.linspace(0, 1, len(centers)))

for i, color in enumerate(colors):
    # Zeichne die Datenpunkte und das zugehörige Clusterzentrum in der gleichen Farbe
    plt.scatter([x for j, x in enumerate(data_x) if cluster_ids[j] == i],
                [y for j, y in enumerate(data_y) if cluster_ids[j] == i],
                color=color, label=f'Cluster {i}')
    # plt.scatter(centers_x[i], centers_y[i], color=color, marker='x')

plt.title('CluStream Cluster Visualisierung')
plt.xlabel('X-Wert')
plt.ylabel('Y-Wert')

# Platzierung der Legende außerhalb des Plots rechts oben
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# Adjust subplot parameters to give the plot more room on the right-hand side
plt.subplots_adjust(right=0.7)
plt.show()