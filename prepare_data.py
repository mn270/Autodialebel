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
FULL_DATA = pd.read_csv(path)
DATA = FULL_DATA["tokens"].tolist()

flair_forward  = FlairEmbeddings('news-forward-fast')
flair_backward = FlairEmbeddings('news-backward-fast')

stacked_embeddings = StackedEmbeddings( embeddings = [
                                                       flair_forward,
                                                       flair_backward
                                                      ])

w_embedding = []
pos = []
feauter_pos = []
Nons_emb = []
Nons_word = []
Nons_vec = []
padding = np.zeros(2048)
for s in tqdm(DATA):
    sentence = Sentence(s)
    sentence.tokens.pop(0)
    sentence.tokens.pop(len(sentence)-1)
    stacked_embeddings.embed(sentence)
    w = []
    for token in sentence:
        w.append(token.embedding.cpu().detach().numpy())
    for t in w:
        lenth = len(w)
        if lenth < 20:
            for i in range(20 - lenth):
              w.append(padding)
        else:
            w = w[:20]
    w = np.array(w)
    w_embedding.append(w)

    token = nltk.word_tokenize(s)
    token = token[1:len(token)-2]
    if len(token) < 20:
        token = token[1:len(tags)-1]
        for i in range(20 - len(token)):
            token.append(' ')
    else:
        token = token[1:21]
    tags = nltk.pos_tag(token)
    counts = Counter(tag for word, tag in tags)
    vector = np.zeros(len(TAGS))
    i=0
    nons = []
    for tag in tags:
        if (tag[1] == 'NN' or tag[1] == 'NNP' or tag[1] == 'NNS' or tag[1] == 'NNPS'):
            Nons_emb.append(w[i])
            nons.append(w[i])
            Nons_word.append(tag[0])
        i+=1
    Nons_vec.append(nons)
    for t in counts:
        id = TAGS.index(t)
        vector[id] = counts[t]
    feauter_pos.append(vector)
    pos.append(tags)
w_embedding = np.array(w_embedding)
feauter_pos = np.array(feauter_pos)
Nons_emb = np.array(Nons_emb)
Nons_word = np.array(Nons_word)
Nons_vec = np.array(Nons_vec)
save('feauter_word_new.npy', w_embedding)
save('feauter_pos_new.npy', feauter_pos)
save('nons_emb_new.npy', Nons_emb)
save('nons_word_new.npy', Nons_word)
save('nons_vec_new.npy', Nons_vec)