import numpy as np
import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, List, Dict, Any, Iterable

from tweet_utils import get_tweets, clean_tweet, CleanedTweet, count_vectorize
from file_utils import read_as_json_gz, write_as_json_gz, write_as_pickle_gz


def get_dataset(tweets_path: str, labeled_data_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    # Create the tweet vectorizer using the full dataset
    tweets = [t.cleaned for t in get_tweets(tweets_path)]
    vectorizer, _ = count_vectorize(tweets, min_df=0.01)

    # Lists to hold inputs and outputs
    X: List[np.ndarray] = []
    y: List[int] = []
    
    # Fetch labeled tweets
    label_counter: Counter = Counter()
    labeled_tweets: Iterable[Dict[str, Any]] = chain(*(read_as_json_gz(path) for path in labeled_data_paths))

    for tweet_dict in labeled_tweets:
        cleaned_tweet: CleanedTweet = clean_tweet(tweet_dict['tweet'])
        input_features = vectorizer.transform([cleaned_tweet.text]).toarray()[0]

        label = int(tweet_dict['label'])
        label_counter[label] += 1

        X.append(input_features)
        y.append(label)

    print('Count distribution: 0 -> {0}, 1 -> {1}, 2 -> {2}'.format(label_counter[0], label_counter[1], label_counter[2]))

    return np.array(X), np.array(y), vectorizer


def plot_confusion_matrix(model: LogisticRegression, X: np.ndarray, y: np.ndarray, output_folder: str):

    # Create the confusion matrix
    y_pred = model.predict(X)
    num_classes = len(model.classes_)
    confusion_mat = np.zeros(shape=(num_classes, num_classes))
    
    for pred, actual in zip(y_pred, y):
        confusion_mat[actual][pred] += 1

    # Plot the confusion matrix
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(confusion_mat, cmap=plt.get_cmap('magma_r'))

        for (i, j), z in np.ndenumerate(confusion_mat):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax.set_title('Confusion Matrix on Test Set')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('True')

        plt.savefig(os.path.join(output_folder, 'confusion_matrix.pdf'))


def evaluate_model(model: BaggingClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X)

    macro_f1 = f1_score(y_true=y, y_pred=y_pred, average='macro')
    micro_f1 = f1_score(y_true=y, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true=y, y_pred=y_pred)

    return {
        'Micro F1': micro_f1,
        'Macro F1': macro_f1,
        'Accuracy': accuracy
    }


def fit_model(X: np.ndarray, y: np.ndarray, n_estimators: int, model_name: str) -> Any:

    model_name = model_name.lower()

    if model_name == 'ada_boost':
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                                   n_estimators=n_estimators)
    elif model_name == 'logistic_regression':
        model = BaggingClassifier(base_estimator=LogisticRegression(C=1.0),
                                  n_estimators=n_estimators)
    elif model_name == 'svm':
        model = BaggingClassifier(base_estimator=SVC(C=1.0, kernel='sigmoid'),
                                  n_estimators=n_estimators)
    else:
        raise ValueError('Unknown model type: {0}'.format(model_name))

    model.fit(X, y)
    return model


def label_dataset(tweets_path: str, model: Any, vectorizer: CountVectorizer, output_file: str):
    
    labeled_dataset: List[Dict[str, Any]] = []
    for tweet in get_tweets(tweets_path):
        features = vectorizer.transform([tweet.cleaned]).toarray()
        label = model.predict(features)[0]

        labeled_dataset.append(dict(tweet=tweet.original, label=int(label), url=tweet.url))

    write_as_json_gz(labeled_dataset, output_file)


if __name__ == '__main__':
    parser = ArgumentParser('Script to classify tweets using logistic regression.')
    parser.add_argument('--tweets-file', type=str, required=True)
    parser.add_argument('--labeled-data-files', type=str, nargs='+')
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--num-estimators', type=int, default=50)
    args = parser.parse_args()

    # Create the dataset
    X, y, vectorizer = get_dataset(args.tweets_file, args.labeled_data_files)

    # Split data into training and validation folds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)

    # Fit and evaluate the model
    model = fit_model(X_train, y_train, n_estimators=args.num_estimators, model_name=args.model_name)

    train_metrics = evaluate_model(model, X=X_train, y=y_train)
    test_metrics = evaluate_model(model, X=X_test, y=y_test)

    print('Train Metrics: {0}'.format(train_metrics))
    print('Test Metrics: {0}'.format(test_metrics))

    # Create the output folder if required
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    # Construct the confusion matrix on the test set
    plot_confusion_matrix(model, X_test, y_test, output_folder=args.output_folder)

    # Use the model to label the full dataset
    label_dataset(args.tweets_file, model, vectorizer, output_file=os.path.join(args.output_folder, 'labeled_dataset.json.gz'))

    # Serialize the model and corresponding results
    model_path = os.path.join(args.output_folder, 'logistic_regression_model.pkl.gz')
    write_as_pickle_gz(model, model_path)

    model_results_file = os.path.join(args.output_folder, 'model_results.json.gz')
    write_as_json_gz(dict(train=train_metrics, test=test_metrics), model_results_file)
