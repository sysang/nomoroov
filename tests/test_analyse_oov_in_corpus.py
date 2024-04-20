from nomoroov.analyse_oov_in_corpus import count_oov
from nomoroov.base_nlp import Token


def test_count_oov():
    sequences = [
        [
            Token(text='hello', is_oov=False),
            Token(text='i', is_oov=False),
            Token(text='am', is_oov=False),
            Token(text='spacy', is_oov=True),
        ],
        [
            Token(text='bonjour', is_oov=True),
            Token(text='i', is_oov=False),
            Token(text='am', is_oov=False),
            Token(text='spacy', is_oov=True),
        ],
        [
            Token(text='bonjour', is_oov=True),
            Token(text=',', is_oov=False),
            Token(text='hola', is_oov=True),
            Token(text='spacy', is_oov=True),
        ],
    ]
    expected = {
        'spacy': 3,
        'bonjour': 2,
        'hola': 1
    }
    actual = count_oov(sequences)

    assert actual == expected
