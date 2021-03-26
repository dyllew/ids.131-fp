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




## Data Loading


# Note that CNN.200910 is empty.  Delete from directory before running function


def get_data(parent):

		## Helper function to sort files into correct dataframes
		def get_channel(file):
    ans = ""
    for char in file:
        if char.isalpha():
            ans += char
        else:
            break
    return ans


    cnn_list = []
		fox_list = []
		msnbc_list = []
		parent = "./TelevisionNews/"
		for file in os.listdir("./TelevisionNews/"):
    		channel = get_channel(file)
    		path = os.path.join(parent, file)
    		if channel=="CNN":
        		cnn_list.append(pd.read_csv(path))
    		elif channel=="FOXNEWS":
        		fox_list.append(pd.read_csv(path))
    		else:
        		msnbc_list.append(pd.read_csv(path))

    cnn = pd.concat(cnn_list)
		fox = pd.concat(fox_list)
		msnbc = pd.concat(msnbc_list)

		return cnn,fox,msnbc




