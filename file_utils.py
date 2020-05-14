import json
import gzip
import pickle
from typing import Any


def write_as_json_gz(data: Any, path: str):
    serialized = json.dumps(data)
    encoded = serialized.encode('utf-8')

    with gzip.GzipFile(path, 'w') as output_file:
        output_file.write(encoded)


def read_as_json_gz(path: str) -> Any:
    with gzip.GzipFile(path, 'r') as input_file:
        data = input_file.read()
        return json.loads(data.decode('utf-8'))


def write_as_pickle_gz(data: Any, path: str):
    with gzip.GzipFile(path, 'wb') as output_file:
        pickle.dump(data, output_file)


def read_as_pickle_gz(path: str) -> Any:
    with gzip.GzipFile(path, 'rb') as input_file:
        return pickle.load(input_file)

