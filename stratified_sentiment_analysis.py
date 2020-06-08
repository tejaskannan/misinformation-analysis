import numpy as np
import scipy.stats as stats
from argparse import ArgumentParser
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from typing import List, Dict

from tweet_utils import get_tweets, count_vectorize, clean_tweet
from file_utils import read_as_json_gz, write_as_json_gz


def sentence_words(text: TextBlob) -> List[int]:
    num_words: List[int] = []
    for sentence in text.sentences:
        num_words.append(len(sentence.words))
    return num_words


def analyze_sentiments(tweets: List[str]) -> Dict[str, List[float]]:
    subjectivities: List[float] = []
    polarities: List[float] = []
    words_per_sentence: List[float] = []
    total_words: List[float] = []
    word_lengths: List[float] = []

    for tweet in tweets:
        text = TextBlob(tweet)

        polarity, subjectivity = text.sentiment
        
        subjectivities.append(subjectivity)
        polarities.append(polarity)

        num_words = sentence_words(text)
        words_per_sentence.extend(num_words)

        total_words.append(len(text.words))

        word_lengths.extend((len(t) for t in text.words))

    return {
        'subjectivity': subjectivities,
        'polarity': polarities,
        'words_per_sentence': words_per_sentence,
        'total_words': total_words,
        'word_lengths': word_lengths
    }


def analyze_text(tweets_file: str, output_file: str):
    # Load data and extract classes
    tweets = read_as_json_gz(tweets_file)

    anti_tweets = [clean_tweet(t['tweet'], should_remove_stopwords=False).text for t in tweets if t['label'] == 1]
    pro_tweets = [clean_tweet(t['tweet'], should_remove_stopwords=False).text for t in tweets if t['label'] == 0]

    anti_sentiments = analyze_sentiments(anti_tweets)
    pro_sentiments = analyze_sentiments(pro_tweets)

    # Compute Averages and Std Dev of each metric, as well as pairwise statistical tests
    anti_avg_results: Dict[str, Dict[str, float]] = dict()
    pro_avg_results: Dict[str, Dict[str, float]] = dict()
    stat_tests: Dict[str, Dict[str, float]] = dict()

    for key in anti_sentiments.keys():
        anti_avg_results[key] = dict(avg=np.average(anti_sentiments[key]), std=np.std(anti_sentiments[key]), median=np.median(anti_sentiments[key]))
        pro_avg_results[key] = dict(avg=np.average(pro_sentiments[key]), std=np.std(pro_sentiments[key]), median=np.median(pro_sentiments[key]))

        t_stat, p_value = stats.ttest_ind(anti_sentiments[key], pro_sentiments[key], equal_var=False)
        stat_tests[key] = dict(t_stat=t_stat, p_value=p_value)

    results = {
        'anti': anti_avg_results,
        'pro': pro_avg_results,
        't_tests': stat_tests
    }
    write_as_json_gz(results, output_file)    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tweets-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    analyze_text(args.tweets_file, args.output_file)
