import nltk
import string
import os
import collections

nltk.download('stopwords')
nltk.download('punkt')

from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Read  html files in path and transform in text
path = '/corpus/html'
token_dict = {}
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        html = shakes.read()
        # Extract text from html
        text = BeautifulSoup(html).get_text().encode('ascii', 'ignore')
        # Lowercase and remove punctuation
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        token_dict[file] = no_punctuation

# Tokenizer function
def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    # text = text.translate(string.punctuation)
    tokens = word_tokenize(text)
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# Transform texts to Tf-Idf coordinates
# (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
vectorizer = TfidfVectorizer(tokenizer=process_text,
                             stop_words=stopwords.words('english'),
                             max_df=0.5,
                             min_df=0.1,
                             lowercase=True)
tfidf_model = vectorizer.fit_transform(token_dict.values())

# Example of a sentence Tf-Idf vectorization
sentence = 'this sentence has seen text such as computer but also animals films or kings'
response = vectorizer.transform([sentence])
feature_names = vectorizer.get_feature_names()
for col in response.nonzero()[1]:
    print feature_names[col], ' - ', response[0, col]

# Cluster texts using K-Means
# (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
km_model = KMeans(n_clusters=3, verbose=0)
km_model.fit(tfidf_model)

# Print clusters
clusters = collections.defaultdict(list)
for idx, label in enumerate(km_model.labels_):
    clusters[label].append(idx)

dict(clusters)

# Print labels of each url webpage
for idx, label in enumerate(km_model.labels_):
    print str(label) + ':', token_dict.keys()[idx].replace('_','/').replace('.html','')
