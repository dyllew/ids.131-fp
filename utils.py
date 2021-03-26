import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import string
import os
import shutil
import re
import nltk
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

# constants
BBC_NEWS_STRING = 'BBCNEWS'
BAD_DF_NAME = 'CNN.200910.csv'
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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
def rm_stopwords(text, stopwords):
    return [i for i in text if not i in stopwords]
    
# lemmatize
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

## Helper function to sort files into correct dataframes
def get_channel(file):
    ans = ""
    for char in file:
        if char.isalpha():
            ans += char
        else:
            break
    return ans

def get_data(parent):
    cnn_list = []
    fox_list = []
    msnbc_list = []
    for file in os.listdir(parent):
        if file == BAD_DF_NAME:
            continue
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

    return cnn, fox, msnbc


# Other
def make_modified_data_folder(src_dir, dest_dir):
    news_folder_path = src_dir
    news_folder_V2_path = dest_dir

    try: 
        os.mkdir(news_folder_V2_path)
    except FileExistsError:
        print('Directory {} already exists'.format(news_folder_V2_path))

    filenames = os.listdir(news_folder_path)
    print('Adding files to {}'.format(dest_dir))
    for file in filenames:
        channel = get_channel(file)
        if file == BAD_DF_NAME or channel == BBC_NEWS_STRING:
            continue
        src_path = os.path.join(news_folder_path, file)
        dest_path = os.path.join(news_folder_V2_path, file)
        # Copy the files we want into the new modified 
        shutil.copy(src_path, dest_path)
    print('All files added to {}'.format(dest_dir))