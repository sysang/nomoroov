import os
import abc
from typing import Iterable


class DatasourceBase(abc.ABC):
    @abc.abstractmethod
    def read(self) -> Iterable[str]:
        pass


class TextFileDatasource(DatasourceBase):
    def __init__(self, dirpath):
        self.dirpath = dirpath

    def read(self) -> Iterable[str]:
        for filename in os.listdir(self.dirpath):
            if not filename.endswith('.txt'):
                continue

            with open(f'{self.dirpath}/{filename}', mode='r', encoding='utf-8') as fd:
                lines = fd.readlines()
                for line in lines:
                    yield line
