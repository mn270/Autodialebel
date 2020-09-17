from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from numpy import load
import collections
import pickle
from numpy import save

feauter_topic = load('encode_topics_new.npy')
feauter_word = load('encode_words_new.npy')
feauter_key = load('encode_key_new.npy')
feauter_pos = load('encode_pos_new.npy')
path = "atis.train.csv"
FULL_DATA = pd.read_csv(path)
INTENT = FULL_DATA["intent"].tolist()
DATA = FULL_DATA["tokens"].tolist()

KEYS = [*dict.fromkeys(INTENT)]

def coefficientmatrix(dict, prediction, data):
    coeff = np.zeros((max(prediction)+1, len(dict)))
    for i in range(len(data)):
        coeff[prediction[i]][dict.index(data[i])]+=1
    count = collections.Counter(data)
    i = 0
    for d in dict:
        coeff[:, i] = coeff[:, i]/count[d]
        i+=1
    coeff = coeff/coeff.max()
    return coeff

X = np.concatenate((feauter_topic, feauter_word), axis=1)
X = np.concatenate((X, feauter_pos), axis=1)
assembled_feauter = np.concatenate((X, feauter_topic), axis=1)
#assembled_feauter = feauter_word

# dendrogram = dendrogram(linkage(X, method = "ward"))
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()
#hc = AgglomerativeClustering(distance_threshold=5, affinity = 'euclidean', linkage ='ward',n_clusters=None)
#hc = AgglomerativeClustering( affinity = 'euclidean', linkage ='ward',n_clusters= 17)
#hc = DBSCAN(eps=0.5, min_samples=15).fit(assembled_feauter)
#hc = KMeans(n_clusters=17)
hc = SpectralClustering(n_clusters=17)
y_hc=hc.fit_predict(assembled_feauter)
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y_hc)
# filename = 'intention_model.sav'
# pickle.dump(neigh, open(filename, 'wb'))
# silhouette_score = metrics.silhouette_score(X, y_hc, metric='euclidean')
# print("Silhouette_score_hier: ")
# print(silhouette_score)

# model = TSNE(n_components=2, random_state=1)
# np.set_printoptions(suppress=True)
# Y = model.fit_transform(X)
# plt.scatter(Y[:, 0], Y[:, 1], c=y_hc, s=300, alpha=.6)
p = coefficientmatrix(KEYS, y_hc, INTENT)
fig, ax = plt.subplots()
im = ax.imshow(p)
ax.set_xticklabels(KEYS)
ax.set_xticks(np.arange(len(KEYS)))
ax.set_yticks(np.arange(max(y_hc)+1))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
DATA = np.array(DATA)
PREDICTION = [DATA, y_hc, INTENT]
PREDICTION = np.array(PREDICTION).transpose()
PREDICTION = PREDICTION[np.argsort(PREDICTION[:, 1])]
for j in range(len(y_hc)):
    print(DATA[j])
    print(y_hc[j])

plt.show()
df = pd.DataFrame({'intent': y_hc})
df.to_csv('intent.csv',index=False)
PREDICTION.tolist()
dp = pd.DataFrame({'Text': PREDICTION[:, 0], 'Predict': PREDICTION[:, 1],'Intent': PREDICTION[:, 2]})
dp.to_csv('compare.csv',index=False)
