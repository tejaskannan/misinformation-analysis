import re
import csv
import numpy as np
from typing import Iterable
from textblob.tokenizers import WordTokenizer
from typing import List, Tuple, FrozenSet
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

LENGTH_THRESHOLD = 3
MAX_WORD_LENGTH = 15
CleanedTweet = namedtuple('CleanedTweet', ['text', 'num_tokens'])
Tweet = namedtuple('Tweet', ['original', 'cleaned', 'url'])
SPECIAL_CHARS = re.compile(r'[^a-zA-Z0-9#]')


def load_stopwords() -> FrozenSet[str]:
    with open('stopwords.txt') as stopwords_file:
        stopwords = set()
        for word in stopwords_file:
            if len(word.strip()) > 0:
                stopwords.add(word.strip())

    return frozenset(stopwords)


STOPWORDS = load_stopwords()


def get_tweets(tweet_file_path: str, should_remove_stopwords: bool = False) -> Iterable[Tweet]:
    with open(tweet_file_path, 'r') as tweet_file:
        reader = csv.reader(tweet_file, delimiter=',', quotechar='"')

        file_iterator = iter(reader)
        next(file_iterator)  # Skip file headers

        for tokens in file_iterator:
            tweet = tokens[6]
            cleaned_tweet = clean_tweet(tweet, should_remove_stopwords)

            if cleaned_tweet.num_tokens >= LENGTH_THRESHOLD:
                yield Tweet(original=tokens[6], cleaned=cleaned_tweet.text, url=tokens[-1])


def should_keep_token(token: str, should_remove_stopwords: bool) -> bool:
    return len(token) > 0 and len(token) <= MAX_WORD_LENGTH and (token not in STOPWORDS or not should_remove_stopwords) and \
            not token.startswith('http') and not token.startswith('www') and not token == '#'


def clean_tweet(tweet: str, should_remove_stopwords: bool = False) -> CleanedTweet:
    # Extract tokens from each tweet
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(tweet, include_punc=True)

    cleaned_tokens: List[str] = []
    for token in tokens:
        t = SPECIAL_CHARS.sub('', token).lower()

        # Substitute the & symbol to standardize text
        if t == 'amp':
            t = 'and'

        # Skip all links and empty strings
        if should_keep_token(t, should_remove_stopwords):
            cleaned_tokens.append(t)  # Lowercase all tokens

    cleaned_tweet = ' '.join(cleaned_tokens)
    return CleanedTweet(text=cleaned_tweet, num_tokens=len(cleaned_tokens))


def tf_idf_vectorize(tweets: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(tweets)
    return vectorizer, features


def count_vectorize(tweets: List[str], min_df: float = 0.0) -> Tuple[CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(min_df=min_df)
    features = vectorizer.fit_transform(tweets)
    return vectorizer, features
