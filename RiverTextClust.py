from river import compose
from river import feature_extraction
from river import metrics
from river import cluster

# corpus dient hier als Datensatz
corpus = [
   {"text":'Donald is the best.',"idd":1, "cluster": 1},
   {"text":'Donald Trump is shit!!!',"idd":2,"cluster": 1},
   {"text":'And this is super unrelated.',"idd":3,"cluster": 2},
   {"text":'I love the USA',"idd":4,"cluster": 3},
   {"text":'USA elections are relevant to the entire world',"idd":5,"cluster": 4},
   {"text":'Who are you going to vote for in this years election?',"idd":6,"cluster": 4}
]

stopwords = [ 'stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is', 'I']

nmi = metrics.NormalizedMutualInfo()
homogeneity = metrics.Homogeneity() # homogen, wenn alle Cluster nur Datenpunkte aus einer einzigen Klasse enthalten

# Pipeline behandelt diese beiden Schritte als eine Einheit, wenn predict_one() und learn_one() aufgerufen werden
model = compose.Pipeline(
    feature_extraction.BagOfWords(lowercase=True, ngram_range=(1, 2), stop_words=stopwords),
    cluster.TextClust(real_time_fading=False, fading_factor=0.001, tgap=100, auto_r=True,
    radius=0.9)
)

# Die Schleife symbolisiert das inkrementelle Lernen über den Datensatz
for x in corpus:
    # predict_one() gibt den Index des Clusters zurück, dem das feature-set zugeordnet wird
    y_pred = model.predict_one(x["text"])
    y = x["cluster"]
    # update() berechnet die Metrik für das aktuelle feature-set
    nmi.update(y, y_pred)
    homogeneity.update(y, y_pred)
    # learn_one() aktualisiert das TextClust-Modell mit dem neuen feature-set
    model.learn_one(x["text"])

print(nmi) # 0.6067810370082493
print(homogeneity) # 58.69%