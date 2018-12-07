# -*- coding: utf-8 -*-

# Author:  Diego Alejandro Asencio Cuellar - diegoasencio96@gmail.com
# Author:  Juan Fonseca

import os, re, operator
from time import time
from bs4 import BeautifulSoup
from string import punctuation
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = stopwords.words('english') + list(punctuation)

ps = PorterStemmer()
vectorizer = TfidfVectorizer()


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

def get_sklearn_tfidf(corpus):
    return vectorizer.fit_transform(corpus)

def get_sklearn_tfidf_test(corpus_test):
    return vectorizer.transform(corpus_test)

print "[+] Inicia el proceso ...."
t0 = time()

path = "../../reuters21578"
files = get_files(path)
corpus = get_corpus(files)
matriz_tfidf = get_sklearn_tfidf(corpus)

tquery = []
query = "Japan to boost its defense spending to help share the burden of protecting Western interests in sensitive areas around the world, including in the Gulf"
tquery.append(" ".join(tokenize(query.lower())))
tquery_tfidf = get_sklearn_tfidf_test(tquery)

rdict = {}
l = len(corpus)
for i in xrange(l):
    sim = cosine_similarity(tquery_tfidf[0].todense(), matriz_tfidf[i].todense())[0][0]
    rdict[sim] = i

result = (sorted(rdict.items(), key=operator.itemgetter(0)))
result = reversed(result)

c = 0
for r in result:
    if c < 10:
        print "[+] Noticia: "+ str(r[1]) + "\n" + corpus[r[1]]+ "\nPorcentaje de similitud: "+str(r[0]*100)+" % \n"
    c+=1

te = time()-t0
print "[+] Tiempo de ejecución: \n" + str(te) + " segundos \n" + str(te/60)+ " minutos"