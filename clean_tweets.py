import pandas as pd
import os
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
import re
from textblob import TextBlob
import numpy as np

class AnalyzeTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self.processTweet(tweet)))
        return processedTweets
    
    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.processTweet(tweet))
        
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', '', tweet) # remove usernames
        tweet = re.sub('-', ' ', tweet) # replaces dashes with a space
        tweet = re.sub(r"[^\w\d'\s]+" ,'',tweet) # removes all punctuation except the single apostrophe
        tweet = re.sub(r'[0-9]+', '', tweet) #removes all numbers
        #tweet = re.sub(' coronavirus ', '', tweet, flags=re.IGNORECASE) # removes coronavirus
        tweet = word_tokenize(tweet) # condenses repeated characters (helloooooooo into hello)
        return ' '.join([word for word in tweet if word not in self._stopwords])

df = pd.read_csv('coronavirus.csv')
tweetanalyzer = AnalyzeTweets()
df['sentiment'] = np.array([tweetanalyzer.analyze_sentiment(tweet) for tweet in df['text']])
df['positive'] = np.array([1 if sentiment==1 else 0 for sentiment in df['sentiment']])
df['neutral'] = np.array([1 if sentiment==0 else 0 for sentiment in df['sentiment']])
df['negative'] = np.array([1 if sentiment==-1 else 0 for sentiment in df['sentiment']])
print(df.head(10))
df.to_csv ('out.csv', index = False, header=True)