from collections import Counter
from typing import Generator


class ConllEntity:
    tokens: list[str]
    labels: list[str]

    def __init__(self, tokens: list[str], labels: list[str]):
        self.tokens = tokens
        self.labels = labels


class ConllReader:
    def read(self, conll_file: str):
        with open(conll_file, 'r', encoding='utf8') as f:
            tokens = []
            labels = []
            for line in f:
                if line == '' or line == '\n':
                    if tokens:
                        yield ConllEntity(tokens, labels)
                        tokens = []
                        labels = []
                else:
                    splits = line.rsplit(' ', 1)
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())
            if tokens:
                yield ConllEntity(tokens, labels)


class MultiConllReader:

    def read(self, conll_file: str):
        reader = ConllReader()
        for entity in reader.read(conll_file):
            yield entity
