from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numpy import load
import pickle

w_embedding = []
pos = []
feauter_pos = []
Nons = []

data = load('nons_emb_new.npy')
nons = load('nons_word_new.npy')
i = 0
index = []
for t in nons:
    if(t == ' '):
        index.append(i)
    i+=1
data = np.delete(data,index,axis=0)
nons = np.delete(nons,index)
X = data


# dendrogram = dendrogram(linkage(X, method  = "ward"))
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()
hc = AgglomerativeClustering(affinity = 'euclidean', linkage ='ward',n_clusters=8)
#hc = AgglomerativeClustering( affinity = 'euclidean', linkage ='ward',n_clusters=14)
y_hc=hc.fit_predict(X)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y_hc)
filename = 'neigh_model.sav'
pickle.dump(neigh, open(filename, 'wb'))
silhouette_score = metrics.silhouette_score(X, y_hc, metric='euclidean')
print("Silhouette_score_hier: ")
print(silhouette_score)

model = TSNE(n_components=2, random_state=1)
np.set_printoptions(suppress=True)
Y = model.fit_transform(X)
plt.scatter(Y[:, 0], Y[:, 1], c=y_hc, s=300, alpha=.6)

for j in range(len(y_hc)):
    print(nons[j])
    print(y_hc[j])

plt.show()


