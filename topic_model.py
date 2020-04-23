import numpy as np
from argparse import ArgumentParser
from sklearn.decomposition import LatentDirichletAllocation

from tweet_utils import get_tweets, count_vectorize


def topic_model(input_path: str, num_topics: int, num_words: int):

    tweets = list(get_tweets(input_path))
    vectorizer, features = count_vectorize(tweets)

    vocab = vectorizer.get_feature_names()

    # Fit the topic model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(features)

    topic_words = dict()
    for index, component in enumerate(lda.components_):
        top_indices = np.argsort(component)[::-1][:num_words]

        topic_tokens = [vocab[i] for i in top_indices]

        print('Topic {0}: {1}'.format(index, ' '.join(topic_tokens)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--num-topics', type=int, required=True)
    parser.add_argument('--num-words', type=int, required=True)
    args = parser.parse_args()

    topic_model(args.input_file, args.num_topics, args.num_words)
