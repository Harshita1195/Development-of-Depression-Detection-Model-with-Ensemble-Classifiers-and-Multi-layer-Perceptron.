import numpy as np
import pandas as pd
import re
import string
from string import punctuation
from collections import Counter

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split

def unseen_text(message):
	df = pd.read_csv("data.csv", usecols=['tweet', 'sentiment'])

	# Return a translation table usable for str.translate()
	df['tweet'] = df['tweet'].str.translate(str.maketrans('','', string.punctuation))

	# function to clean tweets
	def processTweet(tweet):
	    # Remove HTML special entities (e.g. &amp;)
	    tweet = re.sub(r'\&\w*;', '', tweet)
	    #Convert @username to AT_USER
	    tweet = re.sub('@[^\s]+','',tweet)
	    # Remove tickers
	    tweet = re.sub(r'\$\w*', '', tweet)
	    # To lowercase
	    tweet = tweet.lower()
	    # Remove
	    tweet = re.sub(r'\€\w*','', tweet)
	    tweet = re.sub(r'\€™\w*','', tweet)
	    tweet = re.sub(r'\â€¦\w*','', tweet)
	    tweet = re.sub(r'\€¦\w*','', tweet)
	    tweet = re.sub(r'\¦\w*','', tweet)
	    tweet = re.sub(r'\™\w*','', tweet)
	    # Remove hyperlinks
	    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
	    # Remove hashtags
	    tweet = re.sub(r'#\w*', '', tweet)
	    # Remove words with 2 or fewer letters
	    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
	    # Remove whitespace (including new line characters)
	    tweet = re.sub(r'\s\s+', ' ', tweet)
	    # Remove single space remaining at the front of the tweet.
	    tweet = tweet.lstrip(' ') 
	    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
	    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
	    return tweet

	# storing the processed tweets in new feature.
	df['text'] = df['tweet'].apply(processTweet)

	# tokenize function
	def text_process(raw_text):
	    """
	    Takes in a string of text, then performs the following:
	    1. Remove all punctuation
	    2. Remove all stopwords
	    3. Returns a list of the cleaned text
	    """
	    # Check characters to see if they are in punctuation
	    nopunc = [char for char in list(raw_text) if char not in string.punctuation]
	    # Join the characters again to form the string.
	    nopunc = ''.join(nopunc)
	    
	    # Now just remove any stopwords
	    return [word for word in nopunc.lower().split() if word.lower() not in stopwords.words('english')]

	# We split the data into training and testing set (70:30).
	X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.30, random_state=42)

	# vectorize
	bow_transformer = CountVectorizer(analyzer=text_process)

	X_train = bow_transformer.fit_transform(X_train)

	#Term Frequency, Inverse Document Frequency
	from sklearn.feature_extraction.text import TfidfTransformer
	tfidf_transformer = TfidfTransformer()

	X_train = tfidf_transformer.fit_transform(X_train)
	
	tp = processTweet(message)
	test = bow_transformer.transform([tp])
	test = tfidf_transformer.transform(test)

	return test


