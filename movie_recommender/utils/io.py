import json
from pathlib import Path

__all__ = ['load_json', 'dump_json']


def load_json(json_file: Path):
    with open(json_file, 'r') as fp:
        return json.load(fp)


def dump_json(obj, json_file: Path):
    with open(json_file, 'w') as fp:
        json.dump(obj, fp)
