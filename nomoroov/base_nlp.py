import abc
from typing import List, TypeAlias
from dataclasses import dataclass


@dataclass(slots=True)
class Token():
    text: str
    is_oov: bool


@dataclass(slots=True)
class Doc():
    tokens: List[Token]
    text: str


class BaseNlp(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, sequence: str) -> Doc:
        pass
