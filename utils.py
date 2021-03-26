import pandas as pd 
import matplotlib.pyplot as plt 
import string 
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.decomposition import LatentDirichletAllocation, NMF

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


## Preprocessing

# make lowercase
def lowercase(text):
    return text.lower()

# remove numbers
def rm_numbers(text):
    return re.sub(r'\d+', '', text)
    

# remove punctuation
def rm_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# tokenize
def tokenize(text):
    return word_tokenize(text)
    

# remove stopwords
stop_words = set(stopwords.words('english'))
def rm_stopwords(text, stopwords):
    return [i for i in text if not i in stopwords]
    

# lemmatize
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
     return [lemmatizer.lemmatize(token) for token in text]



def preprocessing(text, stopwords,join=False):
    text = lowercase(text)
    text = rm_numbers(text)
    text = rm_punctuation(text)
    text = tokenize(text)
    text = rm_stopwords(text, stopwords)
    text = lemmatize(text)
    if join:
        text = ' '.join(text)
    return text




