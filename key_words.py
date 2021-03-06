import pickle
import pandas as pd
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings
from flair.embeddings import FlairEmbeddings
import numpy as np
import nltk
from collections import Counter
from tqdm import tqdm
from numpy import save

TAGS = [ 'CC', 'CD', 'DT', 'EX' ,'FW', 'IN' ,'JJ' ,'JJR' ,'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN',  'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB' ]
path = "atis.train.csv"
data = pd.read_csv(path)
data_list = data['tokens'].tolist()
filename = 'neigh_model_2.sav'
neigh = pickle.load(open(filename, 'rb'))

flair_forward  = FlairEmbeddings('news-forward-fast')
flair_backward = FlairEmbeddings('news-backward-fast')

stacked_embeddings = StackedEmbeddings( embeddings = [
                                                       flair_forward,
                                                       flair_backward
                                                      ])

key_words = []
clusters = 14
for s in tqdm(data_list):
    sentence = Sentence(s)
    stacked_embeddings.embed(sentence)
    w = []
    for token in sentence:
        w.append(token.embedding.cpu().detach().numpy())
    w = np.array(w)
    token = nltk.word_tokenize(s)
    tags = nltk.pos_tag(token)
    counts = Counter(tag for word, tag in tags)
    vector = np.zeros(len(TAGS))
    i=0
    nons = []
    vect = np.zeros(clusters)
    for tag in tags:
        if (tag[1] == 'NN' or tag[1] == 'NNP' or tag[1] == 'NNS' or tag[1] == 'NNPS'):
            k = np.reshape(w[i], (1, -1))
            l = neigh.predict(k)
            vect[l]+=1
            o = vect[l]
        i+=1
    key_words.append(vect)

key_words = np.array(key_words)
save('key_words_new.npy', key_words)