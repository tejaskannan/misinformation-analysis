from argparse import ArgumentParser
from collections import Counter
from textblob.tokenizers import WordTokenizer
from typing import List

from tweet_utils import clean_tweet, count_vectorize
from file_utils import read_as_json_gz


def find_ngrams(tweets: List[str], n: int, top: int):
    
    ngram_counter: Counter = Counter()

    for tweet in tweets:
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize(tweet, include_punc=True)

        for i in range(len(tokens) - n):
            subwords = ' '.join(tokens[i:i+n])
            ngram_counter[subwords] += 1

    print(ngram_counter.most_common(top))


def find_top_ngrams(data_file: str, n: int, top: int):
    # Load data
    tweets = read_as_json_gz(data_file)

    # Separate tweets by class
    anti_tweets = list(set(clean_tweet(t['tweet'], should_remove_stopwords=True).text for t in tweets if t['label'] == 0))
    pro_tweets = list(set(clean_tweet(t['tweet'], should_remove_stopwords=True).text for t in tweets if t['label'] == 1))

    # Fit a topic model for each class
    print('===== Anti 5G - COVID Topics =====')
    find_ngrams(anti_tweets, n=n, top=top)

    print()
    print('===== Pro 5G - COVID Topics =====')
    find_ngrams(pro_tweets, n=n, top=top)


if __name__ == '__main__':
    parser = ArgumentParser('Script to find top word n-grams in each type of tweet')
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--top', type=int, required=True)
    args = parser.parse_args()

    find_top_ngrams(args.data_file, args.n, args.top)
