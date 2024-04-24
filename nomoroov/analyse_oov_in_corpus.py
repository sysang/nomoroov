from typing import List, Iterable, Callable

from .base_nlp import BaseNlp, Token, Doc
from .spacy_nlp import SpacyNlp
from .type_helpers import get_dict_value
from .datasource import DatasourceBase


def count_oov(docs: Iterable[Doc]) -> dict[str, int]:
    oov_data: dict[str, int] = {}
    for doc in docs:
        for token in doc.tokens:
            if not token.is_oov:
                continue
            if oov_data.get(token.text, False):
                oov_data[token.text] += 1
            else:
                oov_data[token.text] = 1

    sorted_keys = sorted(
        oov_data.keys(), key=get_dict_value(oov_data), reverse=True)
    sorted_oov_data: dict[str, int] = {}
    for k in sorted_keys:
        sorted_oov_data[k] = oov_data[k]

    return sorted_oov_data

