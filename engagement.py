import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict

from file_utils import read_as_json_gz, write_as_json_gz


Engagement = namedtuple('Engagement', ['retweets', 'favorites', 'replies'])


def create_engagement_dict(path: str) -> Dict[str, Engagement]:
    result: Dict[str, Engagement] = dict()

    with open(path, 'r') as fin:
        reader = csv.reader(fin, quotechar='|', delimiter=',')

        file_iter = iter(reader)
        next(file_iter)  # Skip the headers

        for record in file_iter:
            url = record[-1]
            replies, retweets, favorites = int(record[3]), int(record[4]), int(record[5])

            result[url] = Engagement(retweets=retweets, favorites=favorites, replies=replies)

    return result


def compare_engagement(label_path: str, engagement_dict: Dict[str, Engagement], output_file: str):
    tweets = read_as_json_gz(label_path)

    anti_urls = [t['url'] for t in tweets if t['label'] == 1]
    pro_urls = [t['url'] for t in tweets if t['label'] == 0]

    anti_engagement = [engagement_dict[url] for url in anti_urls]
    pro_engagement = [engagement_dict[url] for url in pro_urls]

    results: Dict[str, Dict[str, float]] = dict()
    for field in Engagement._fields:
        anti_values = [e._asdict()[field] for e in anti_engagement]
        pro_values = [e._asdict()[field] for e in pro_engagement]

        anti_results = {
            'average': float(np.average(anti_values)),
            'median': float(np.median(anti_values)),
            'std': float(np.std(anti_values)),
            'max': float(np.max(anti_values)),
            'min': float(np.min(anti_values))
        }
        
        pro_results = {
            'average': float(np.average(pro_values)),
            'median': float(np.median(pro_values)),
            'std': float(np.std(pro_values)),
            'max': float(np.max(pro_values)),
            'min': float(np.min(pro_values))
        }

        stat, p_value = stats.ttest_ind(anti_values, pro_values, equal_var=False)
        results[field] = dict(anti=anti_results, pro=pro_results, test=dict(stat=stat, p_value=p_value))

    write_as_json_gz(results, output_file)

    # Print out the results in a Latex-like format for convenience
    labels = ['min', 'median', 'max', 'average', 'std']
    for field, result_dicts in results.items():
        print(field)
        
        anti_results = result_dicts['anti']
        row = ['{0:.3f}'.format(anti_results[label]) for label in labels]
        print('Anti: ')
        print(' & '.join(row))

        pro_results = result_dicts['pro']
        row = ['{0:.3f}'.format(pro_results[label]) for label in labels]
        print('Pro: ')
        print(' & '.join(row))

        print('==========')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tweets-file', type=str, required=True)
    parser.add_argument('--label-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    engagement_dict = create_engagement_dict(args.tweets_file)
    compare_engagement(args.label_file, engagement_dict, args.output_file)
