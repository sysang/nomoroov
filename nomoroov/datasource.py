import os
import abc
from typing import Iterable

from .spacy_nlp import BaseNlp, Doc


class DatasourceBase(abc.ABC):
    @abc.abstractmethod
    def read(self) -> Iterable[Doc]:
        pass


class TextFileDatasource(DatasourceBase):
    def __init__(self, dirpath: str, nlp: BaseNlp):
        self.dirpath = dirpath
        self.nlp = nlp

    def read(self) -> Iterable[Doc]:
        for filename in os.listdir(self.dirpath):
            if not filename.endswith('.txt'):
                continue

            with open(f'{self.dirpath}/{filename}', mode='r', encoding='utf-8') as fd:
                lines = fd.readlines()
                for line in lines:
                    yield self.nlp.tokenize(line)
