import os, re
from bs4 import BeautifulSoup
from string import punctuation
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import scipy.sparse

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
        w = ps.stem(w)
        if w not in stop_words and not w.isdigit() and len(str(w)) > 3:
            aux.append(w)
    return aux
    #return [w for w in words if w not in stop_words and not w.isdigit() and len(str(w))>3]


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

def get_sklearn_dict():
    return vectorizer.get_feature_names()

def get_sklearn_tfidf(corpus):
    return vectorizer.fit_transform(corpus)

def get_sklearn_tfidf_test(corpus_test):
    return vectorizer.transform(corpus_test)

def compare_cosine(d, q):
    return float(dot(d, q) / float(norm(d) * norm(q)))


path = "../../reuters21578"
files = get_files(path)
corpus = get_corpus(files)
matriz_tfidf = get_sklearn_tfidf(corpus)
#print()
#print matriz_tfidf.shape
#cx = scipy.sparse.coo_matrix(matriz_tfidf)
#for i, j, v in zip(cx.row, cx.col, cx.data):
#    print "(%d, %d), %s" % (i, j, v)

#dict = get_sklearn_dict()
#print dict
#print len(dict)

tquery = []

query = "new york house work pay in dollars"

tquery.append(" ".join(tokenize(query.lower())))
tquery_tfidf = get_sklearn_tfidf_test(tquery)
#print tquery_tfidf.shape

results = cosine_similarity(tquery_tfidf[0], matriz_tfidf)
#results.sort(reverse=True)
print results
'''
ratings = []
for documentVector in matriz_tfidf:
    for queryVector in tquery_tfidf:
        ratings.append(compare_cosine(documentVector, queryVector))

#ratings = [compare_cosine(documentVector, tquery_tfidf) for documentVector in matriz_tfidf]
ratings.sort(reverse=True)
print ratings
'''