from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from nltk.cluster import KMeansClusterer
import nltk
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings
from flair.embeddings import FlairEmbeddings
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics
import scipy.cluster.hierarchy as sch
from sklearn.neighbors import KNeighborsClassifier
from numpy import load
import pickle
from nltk.corpus import stopwords

# training data
sentences = ['this is the good car',
             'this is another snow',
             'one  more auto',
             'this is the new bike',
             'this is about winter',
             'and this is snowy']



# sentences = ['this is the good machine learning book',
#              'this is another book',
#              'one  more book',
#              'this is the new post',
#              'this is about machine learning post',
#              'and this is  the last post']

flair_forward  = FlairEmbeddings('news-forward-fast')
flair_backward = FlairEmbeddings('news-backward-fast')

stacked_embeddings = StackedEmbeddings( embeddings = [
                                                       flair_forward,
                                                       flair_backward
                                                      ])

w_embedding = []
pos = []
feauter_pos = []
Nons = []

for s in sentences:
    sentence = Sentence(s)
    stacked_embeddings.embed(sentence)
    w = []
    for token in sentence:
        w_embedding.append(token.embedding.cpu().detach().numpy())



sentence = Sentence('this is another snow or new word')
stacked_embeddings.embed(sentence)
to_predict = []
for token in sentence:
    to_predict.append(token.embedding.cpu().detach().numpy())
# training model
# model = Word2Vec(sentences, min_count=1)
#
# # get vector data
# X = model[model.wv.vocab]
# print(X)
#
# print(model.similarity('this', 'is'))
#
# print(model.similarity('post', 'book'))
#
# print(model.most_similar(positive=['machine'], negative=[], topn=2))
#
# print(model['the'])
#
# print(list(model.wv.vocab))
#
# print(len(list(model.wv.vocab)))
data = load('nons_emb.npy')
nons = load('nons_word.npy')
X = data
# NUM_CLUSTERS = 13
# kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
# assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
# print(assigned_clusters)

# words = list(model.wv.vocab)
# for i, word in enumerate(words):
#     print(word + ":" + str(assigned_clusters[i]))

# kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
# kmeans.fit(X)
# Y = kmeans.predict(X)
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_

#plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='viridis')
dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
hc = AgglomerativeClustering(distance_threshold=5, affinity = 'euclidean', linkage ='ward',n_clusters=None)
y_hc=hc.fit_predict(X)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y_hc)
filename = 'neigh_model.sav'
pickle.dump(neigh, open(filename, 'wb'))

# print(neigh.predict(to_predict))
#
#
#
# print("Cluster id labels for inputted data")
# print(labels)
# print("Centroids data")
# print(centroids)
#
# print(
#     "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
# print(kmeans.score(X))
#
# silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

# print("Silhouette_score: ")
# print(silhouette_score)
silhouette_score = metrics.silhouette_score(X, y_hc, metric='euclidean')
print("Silhouette_score_hier: ")
print(silhouette_score)
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=1)
np.set_printoptions(suppress=True)

Y = model.fit_transform(X)

plt.scatter(Y[:, 0], Y[:, 1], c=y_hc, s=300, alpha=.6)

for j in range(len(y_hc)):
    print(nons[j])
    print(y_hc)

# for j in range(len(sentences)):
#     plt.annotate(labels[j], xy=(Y[j][0], Y[j][1]), xytext=(0, 0), textcoords='offset points')
#     print("%s %s" % (assigned_clusters[j], sentences[j]))

plt.show()


