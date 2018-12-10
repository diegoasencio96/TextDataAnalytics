# -*- coding: utf-8 -*-

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause
# Modificado por: Diego Alejandro Asencio Cuellar
#                   diegoasencio96@gmail.com

from __future__ import print_function
from time import time
import os, re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

corpusg = []


def clean_return(text):
    text = text.lower().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\a', ' ')
    text = text.replace("\'s", "")
    text = re.sub("[^\sa-zA-Z]+", ' ', text)
    return text

def tokenize(text):
    text = clean_return(text)
    words = word_tokenize(text)
    aux = []
    for w in words:
        #w = ps.stem(w)
        if w not in stop_words and not w.isdigit() and len(str(w)) > 3:
            aux.append(w)
    return aux

def get_files(path_analyze):
    path = os.walk(path_analyze)
    files_ = []
    for root, dirs, files in path:
        for fichero in files:
            (name, extension) = os.path.splitext(fichero)
            if (extension == "."+"sgm"):
                files_.append("../../reuters21578" + "/" +name + extension)
    return sorted(files_)


def get_corpus(files):
    corpus = []
    for file in files:
        arch = open(file, 'r')
        text = arch.read()
        arch.close()
        soup = BeautifulSoup(text, "html.parser")
        file_notices = soup.find_all('reuters')
        for notice in file_notices:
            title = notice.find('title')
            body = notice.find('body')
            notice_ = (str(title)[7:-8])+" "+(str(body)[6:-7])
            if not str(notice_) == "":
                corpusg.append(notice_)
            filtered_notices = tokenize(notice_.lower())
            if not str(filtered_notices) == "":
                corpus.append(" ".join(filtered_notices))
    return corpus





t0 = time()

print("[+] Loading dataset and extracting TF-IDF features...")
#dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

#print(dataset.data[0])

path = "../../reuters21578"
files = get_files(path)
corpus = get_corpus(files)


vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
#tfidf = vectorizer.fit_transform(dataset.data[:n_samples])
tfidf = vectorizer.fit_transform(corpus)
print("[+] done in %0.3fs." % (time() - t0))


print(tfidf)


# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
print("done in %0.3fs." % (time() - t0))


# Fit the NMF model
print("[+] Fitting the NMF model with n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
print("[+] done in %0.3fs." % (time() - t0))



temas = []
feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("[+] Topic #%d:" % topic_idx)
    temas = (" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print (temas)
    print()


print("done in %0.3fs." % (time() - t0))


W = nmf.fit_transform(tfidf)
H = nmf.components_
print (W.shape)
print (H.shape)

'''
import scipy.sparse
cx = scipy.sparse.coo_matrix(W)
for i, j, d in zip(cx.col, cx.row, cx.data):
    print (("%d, %d, %s") % (i, j, d))
'''
rdict = {}
po = 0
for i in range(int(W.shape[1])):
    ma = 0.0
    for j in range(int(W.shape[0])):
        valor = (W[j][i])
        if valor >= ma:
            ma = valor
            po = j
    #print(ma, po)said company corp unit contract reuter president agreement systems subsidiary group acquisition chairman chief executive products computer officer sale sell


    print ("Noticia x Topic # "+str(i))
    print(corpusg[po], valor)
    print("")
        #max1.append(valor)
    #sim = (max(max1))
    #print(sim)
#  bank banks rate bond market money debt said issue central loan rates lead manager dollar credit coupon dealers bills
# eurobond



    #rdict[sim] = j


'''
for topic_idx, topic in enumerate(W):
    print("[+] Noticias #%d:" % topic_idx)
    print(topic)
    #print(" ".join([temas[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
'''