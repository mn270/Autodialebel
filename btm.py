import numpy as np
import pyLDAvis
from biterm.cbtm import oBTM
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from numpy import save
import pandas as pd
from sklearn.feature_extraction import text

if __name__ == "__main__":
    path = "atis.train.csv"
    data = pd.read_csv(path)
    data_list = data['tokens'].tolist()

    # vectorize texts
    my_additional_stop_words = ['BOS', 'EOS','bos','eos']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    vec = CountVectorizer(stop_words=stop_words)
    X = vec.fit_transform(data_list).toarray()

    # get vocabulary
    vocab = np.array(vec.get_feature_names())

    # get biterms
    biterms = vec_to_biterms(X)

    # create btm
    btm = oBTM(num_topics=20, V=vocab)

    print("\n\n Train BTM ..")
    topics = btm.fit_transform(biterms, iterations=100)

    print("\n\n Visualize Topics ..")
    vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    pyLDAvis.save_html(vis, '/home/pracownik/PycharmProjects/simple_btm.html')

    print("\n\n Topic coherence ..")
    topic_summuary(btm.phi_wz.T, X, vocab, 10)

    print("\n\n Texts & Topics ..")
    for i in range(len(data_list)):
        print("{} (topic: {})".format(data_list[i], topics[i].argmax()))
    save('topics_new.npy', topics)