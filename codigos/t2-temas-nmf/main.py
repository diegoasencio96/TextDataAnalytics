# -*- coding: utf-8 -*-

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause
# Modificado por: Diego Alejandro Asencio Cuellar
#                   diegoasencio96@gmail.com

from __future__ import print_function
from time import time
import os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20


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


# Fit the NMF model
print("[+] Fitting the NMF model with n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
print("[+] done in %0.3fs." % (time() - t0))




feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("[+] Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


