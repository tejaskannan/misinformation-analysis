import numpy as np
from argparse import ArgumentParser
from textblob import TextBlob
from collections import defaultdict, namedtuple
from sklearn.cluster import KMeans
from typing import Dict, Tuple

from tweet_utils import get_tweets


SentimentTuple = namedtuple('SentimentTuple', ['text', 'polarity', 'subjectivity'])


def cluster(sample_dict: Dict[int, SentimentTuple], mode: str):
    mode = mode.lower()
    assert mode in ('subjectivity', 'polarity'), 'Mode must be one of: subjectivity, polarity'

    scores = [value._asdict()[mode] for value in sample_dict.values()]

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(np.reshape(scores, (-1, 1)))
    centers = kmeans.cluster_centers_

    cluster_tweets = defaultdict(list)

    for element in sample_dict.values():
        score = element._asdict()[mode]
        cluster_id = kmeans.predict([[score]])[0]
        cluster_tweets[cluster_id].append(element.text)

    for cluster_id, tweets in sorted(cluster_tweets.items()):
        print('Cluster ID: {0}, Cluster Center: {1}, Number of Tweets: {2}'.format(cluster_id, centers[cluster_id], len(tweets))) 


def analyze_sentiments(input_file: str):
    
    sample_dict: Dict[int, SentimentTuple] = dict()
    polarities: List[float] = []
    subjectivities: List[float] = []

    for index, tweet in enumerate(get_tweets(input_file)):
        text = TextBlob(tweet)
        sentiment = text.sentiment

        sample_dict[index] = SentimentTuple(text=tweet, polarity=sentiment.polarity, subjectivity=sentiment.subjectivity)
        polarities.append(sentiment.polarity)
        subjectivities.append(sentiment.subjectivity)

    print('==== Sentiment Statistics ====')
    print('Average polarity: {0}'.format(np.average(polarities)))
    print('Std polarity: {0}'.format(np.std(polarities)))
    print('Average subjectivity: {0}'.format(np.average(subjectivities)))
    print('Std subjectivity: {0}'.format(np.std(subjectivities)))
    print('==============================')

    print('==== Clustering Based On Polarity ====')
    cluster(sample_dict, mode='polarity')
    print('======================================')

    print('==== Clustering Based on Subjectivity ====')
    cluster(sample_dict, mode='subjectivity')
    print('==========================================')

 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    args = parser.parse_args()

    analyze_sentiments(args.input_file)
