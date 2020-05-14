from argparse import ArgumentParser
from typing import List, Dict, Any

from tweet_utils import Tweet, get_tweets
from file_utils import write_as_json_gz


def label_data(tweets_file: str, output_file: str, num_samples: int, start_index: int):
    
    dataset: List[Dict[str, Any]] = []

    for tweet_id, tweet in enumerate(get_tweets(tweets_file)):
        if tweet_id < start_index:
            continue

        if len(dataset) >= num_samples:
            break

        print('Tweet {0}: {1}'.format(tweet_id, tweet.original))
        
        while True:
            try:
                print('Enter a label: ', end=' ')
                label = int(input().strip())
                break
            except ValueError:
                pass

        if label in (0, 1):
            dataset.append(dict(tweet=tweet.original, label=label, url=tweet.url))
            print('Dataset Size: {0}'.format(len(dataset)))
        
        print('==========') 

    write_as_json_gz(dataset, output_file)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tweets-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--start-index', type=int, default=0)
    args = parser.parse_args()

    label_data(args.tweets_file, args.output_file, args.num_samples, args.start_index)
