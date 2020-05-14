import numpy as np
import os.path
from argparse import ArgumentParser
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import Tuple, List, Dict, Any

from tweet_utils import get_tweets, clean_tweet, CleanedTweet, count_vectorize
from file_utils import read_as_json_gz, write_as_json_gz, write_as_pickle_gz


def get_dataset(tweets_path: str, labeled_data_path: str) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    # Create the tweet vectorizer using the full dataset
    tweets = [t.cleaned for t in get_tweets(tweets_path)]
    vectorizer, _ = count_vectorize(tweets, min_df=0.01) 

    # Lists to hold inputs and outputs
    X: List[np.ndarray] = []
    y: List[int] = []
    
    # Fetch labeled tweets
    label_counter: Counter = Counter()
    labeled_tweets: List[Dict[str, Any]] = read_as_json_gz(labeled_data_path)

    for tweet_dict in labeled_tweets:
        cleaned_tweet: CleanedTweet = clean_tweet(tweet_dict['tweet'])
        input_features = vectorizer.transform([cleaned_tweet.text]).toarray()[0]

        label = int(tweet_dict['label'])
        label_counter[label] += 1

        X.append(input_features)
        y.append(label)

    print('Count distribution: 0 -> {0}, 1 -> {1}'.format(label_counter[0], label_counter[1]))

    return np.array(X), np.array(y), vectorizer


def fit_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0)
    model.fit(X, y)
    return model


def label_dataset(tweets_path: str, model: LogisticRegression, vectorizer: CountVectorizer, output_file: str):
    
    labeled_dataset: List[Dict[str, Any]] = []
    for tweet in get_tweets(tweets_path):
        features = vectorizer.transform([tweet.cleaned]).toarray()
        label = model.predict(features)[0]

        labeled_dataset.append(dict(tweet=tweet.original, label=int(label), url=tweet.url))

    write_as_json_gz(labeled_dataset, output_file)


if __name__ == '__main__':
    parser = ArgumentParser('Script to classify tweets using logistic regression.')
    parser.add_argument('--tweets-file', type=str, required=True)
    parser.add_argument('--labeled-data-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    # Create the dataset
    X, y, vectorizer = get_dataset(args.tweets_file, args.labeled_data_file)

    # Split data into training and validation folds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)

    # Fit and evaluate the model
    model = fit_model(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print('Train accuracy: {0:.4f}'.format(train_acc))
    print('Test accuracy: {0:.4f}'.format(test_acc))

    # Create the output folder if required
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    # Use the model to label the full dataset
    label_dataset(args.tweets_file, model, vectorizer, output_file=os.path.join(args.output_folder, 'labeled_dataset.json.gz'))

    # Serialize the model and corresponding results
    model_path = os.path.join(args.output_folder, 'logistic_regression_model.pkl.gz')
    write_as_pickle_gz(model, model_path)

    model_results_file = os.path.join(args.output_folder, 'model_results.json.gz')
    write_as_json_gz(dict(train=train_acc, test=test_acc), model_results_file)
