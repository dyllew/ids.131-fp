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
CNN_STRING = 'CNN'
MSNBC_STRING = 'MSNBC'
FOX_NEWS_STRING = 'FOXNEWS'
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

def make_list_of_token_lists(df, stopwords):
    list_of_token_lists = df["Snippet"].apply(lambda x: preprocessing(x, stopwords, join=False)).tolist()
    return list_of_token_lists

def make_list_of_processed_snippets(df, stopwords):
    list_of_processed_snippets = df["Snippet"].apply(lambda x: preprocessing(x, stopwords, join=True)).tolist()
    return list_of_processed_snippets

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
        if channel == CNN_STRING:
            cnn_list.append(pd.read_csv(path))
        elif channel == FOX_NEWS_STRING:
            fox_list.append(pd.read_csv(path))
        else:
            msnbc_list.append(pd.read_csv(path))

    cnn = pd.concat(cnn_list)
    fox = pd.concat(fox_list)
    msnbc = pd.concat(msnbc_list)
    # parse strings into date objects
    for df in [cnn, fox, msnbc]:
        df["DateTime"] = pd.to_datetime(df["MatchDateTime"], format='%m/%d/%Y %H:%M:%S', errors='ignore')
        df.drop(columns=['MatchDateTime'], inplace=True)
    return cnn, fox, msnbc

def make_all_data_df(data_dir):
    cnn, fox, msnbc = get_data(data_dir)
    full_data_df = pd.concat([cnn, fox, msnbc], ignore_index=True)
    return full_data_df


## DataFrame Filtering

# in time
def get_data_between_dates(df, start_date, end_date):
    # start_date and end_date should be strings of the form '2009-12-31' i.e. 'year-month-day'
    # note: works the same for '2009-01-01' and '2009-1-1'
    date_mask = (df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)
    return df[date_mask]

def get_data_by_year(df, year):
    # year is an int
    start_date = '{}-1-1'.format(year)
    end_date = '{}-12-31'.format(year)
    return get_data_between_dates(df, start_date, end_date)

# By channel / network
def get_data_by_channel(df, channel_name):
    # channel_name must be one of CNN_STRING, FOX_NEWS_STRING, MSNBC_STRING
    return df[df['Station'] == channel_name]

# By show
def get_data_by_show(df, show_name):
    return df[df['Show'] == show_name]

## Other useful functions
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