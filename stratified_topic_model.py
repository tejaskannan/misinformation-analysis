import numpy as np
from argparse import ArgumentParser
from sklearn.decomposition import LatentDirichletAllocation
from typing import List

from tweet_utils import get_tweets, count_vectorize, clean_tweet
from file_utils import read_as_json_gz


def fit_topic_model(tweets: List[np.ndarray], n_components: int, n_words: int, vocab: List[str]):
    lda = LatentDirichletAllocation(n_components)
    lda.fit(tweets)

    for index, component in enumerate(lda.components_):
        top_indices = np.argsort(component)[::-1][:n_words]
        topic_words = [vocab[i] for i in top_indices]

        print('Topic {0}: {1}'.format(index, ' '.join(topic_words)))


def get_topics(data_file: str, n_components: int, n_words: int):
    # Load data and fit vectorizer
    tweets = read_as_json_gz(data_file)
    cleaned_tweets = [clean_tweet(t['tweet'], should_remove_stopwords=True).text for t in tweets]
    vectorizer, features = count_vectorize(cleaned_tweets)
    vocab = vectorizer.get_feature_names()

    # Separate tweets by class
    anti_tweets = [vec.toarray().reshape(-1) for vec, tweet in zip(features, tweets) if tweet['label'] == 0]
    pro_tweets = [vec.toarray().reshape(-1) for vec, tweet in zip(features, tweets) if tweet['label'] == 1]

    # Fit a topic model for each class
    print('===== Anti 5G - COVID Topics =====')
    fit_topic_model(anti_tweets, n_components, n_words, vocab=vocab)

    print()
    print('===== Pro 5G - COVID Topics =====')
    fit_topic_model(pro_tweets, n_components, n_words, vocab=vocab)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--n-components', type=int, required=True)
    parser.add_argument('--n-words', type=int, required=True)
    args = parser.parse_args()

    get_topics(args.data_file, args.n_components, args.n_words)
