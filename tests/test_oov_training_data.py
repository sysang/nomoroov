import pytest

from nomoroov.oov_training_data import (
    cook_training_data,
    InMemoryDataCollector,
    filter_data_by_targeted_oov
)
from nomoroov.base_nlp import Doc, Token


test_parameters = [
    (
        iter(range(1, 5)),
        5,
        set([(1, 2), (2, 3), (3, 4), (2, 4), (1, 3), (1, 4)])
    ),
    (
        iter(range(1, 6)),
        5,
        set([(1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (2, 4), (2, 5), (1, 3), (1, 4), (1, 5)])
    ),
    (
        iter(range(1, 7)),
        5,
        set([(1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (2, 4), (2, 5), (1, 3), (1, 4), (1, 5)]),
    ),
    (
        iter(range(1, 8)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (2, 4), (2, 5), (1, 3), (1, 4), (1, 5),
                (6, 7)
            ]
        ),
    ),
    (
        iter(range(1, 9)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (2, 4), (2, 5), (1, 3), (1, 4), (1, 5),
                (6, 7), (6, 8), (7, 8)
            ]
        ),
    ),
    (
        iter(range(1, 10)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5),
                (3, 5),
                (2, 4), (2, 5),
                (1, 3), (1, 4), (1, 5),
                (6, 7), (7, 8), (8, 9),
                (7, 9), 
                (6, 8), (6, 9)
            ]
        ),
    ),
    (
        iter(range(1, 11)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5),
                (3, 5),
                (2, 4), (2, 5),
                (1, 3), (1, 4), (1, 5),
                (6, 7), (7, 8), (8, 9), (9, 10),
                (8, 10),
                (7, 9), (7, 10),
                (6, 8), (6, 9), (6, 10),
            ]
        ),
    ),
    (
        iter(range(1, 12)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5),
                (3, 5),
                (2, 4), (2, 5),
                (1, 3), (1, 4), (1, 5),
                (6, 7), (7, 8), (8, 9), (9, 10),
                (8, 10),
                (7, 9), (7, 10),
                (6, 8), (6, 9), (6, 10),
            ]
        ),
    ),
    (
        iter(range(1, 16)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5),
                (3, 5),
                (2, 4), (2, 5),
                (1, 3), (1, 4), (1, 5),
                (6, 7), (7, 8), (8, 9), (9, 10),
                (8, 10),
                (7, 9), (7, 10),
                (6, 8), (6, 9), (6, 10),
                (11, 12), (12, 13), (13, 14), (14, 15),
                (13, 15),
                (12, 14), (12, 15),
                (11, 13), (11, 14), (11, 15),
            ]
        ),
    ),
    (
        iter(range(1, 17)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5),
                (3, 5),
                (2, 4), (2, 5),
                (1, 3), (1, 4), (1, 5),
                (6, 7), (7, 8), (8, 9), (9, 10),
                (8, 10),
                (7, 9), (7, 10),
                (6, 8), (6, 9), (6, 10),
                (11, 12), (12, 13), (13, 14), (14, 15),
                (13, 15),
                (12, 14), (12, 15),
                (11, 13), (11, 14), (11, 15),
            ]
        ),
    ),
    (
        iter(range(1, 20)),
        5,
        set(
            [
                (1, 2), (2, 3), (3, 4), (4, 5),
                (3, 5),
                (2, 4), (2, 5),
                (1, 3), (1, 4), (1, 5),
                (6, 7), (7, 8), (8, 9), (9, 10),
                (8, 10),
                (7, 9), (7, 10),
                (6, 8), (6, 9), (6, 10),
                (11, 12), (12, 13), (13, 14), (14, 15),
                (13, 15),
                (12, 14), (12, 15),
                (11, 13), (11, 14), (11, 15),
                (16, 17), (17, 18), (18, 19),
                (17, 19),
                (16, 18), (16, 19)
            ]
        ),
    ),
]


@pytest.mark.parametrize("list_of_item,window_size,expected", test_parameters)
def test_cook_training_data(list_of_item, window_size, expected):
    data_collector = InMemoryDataCollector()

    cook_training_data(list_of_item, data_collector=data_collector, window_size=window_size)

    assert expected == set(data_collector.data)


def test_filter_data_by_targeted_oov():
    sequences = [
        Doc(
            text='unit test unit test unit test',
            tokens=[
                Token(text='unit', is_oov=False),
                Token(text='test', is_oov=False),
                Token(text='unit', is_oov=False),
                Token(text='test', is_oov=False),
                Token(text='unit', is_oov=False),
                Token(text='test', is_oov=False),
            ],
        ),
        Doc(
            text='hello world, how are you',
            tokens=[
                Token(text='hello', is_oov=False),
                Token(text='world', is_oov=False),
                Token(text=',', is_oov=False),
                Token(text='how', is_oov=False),
                Token(text='are', is_oov=False),
                Token(text='you', is_oov=False),
            ],
        ),
        Doc(
            text='bonjour i am spacy how are you',
            tokens=[
                Token(text='bonjour', is_oov=True),
                Token(text='i', is_oov=False),
                Token(text='am', is_oov=False),
                Token(text='spacy', is_oov=True),
                Token(text='how', is_oov=False),
                Token(text='are', is_oov=False),
                Token(text='you', is_oov=False),
            ],
        ),
        Doc(
            text='I am spacy, how are you',
            tokens=[
                Token(text='i', is_oov=False),
                Token(text='am', is_oov=False),
                Token(text='spacy', is_oov=True),
                Token(text=',', is_oov=False),
                Token(text='how', is_oov=False),
                Token(text='are', is_oov=False),
                Token(text='you', is_oov=False),
            ],
        ),
        Doc(
            text='hello i am spacy',
            tokens=[
                Token(text='hello', is_oov=False),
                Token(text='i', is_oov=False),
                Token(text='am', is_oov=False),
                Token(text='spacy', is_oov=False),
            ],
        ),
        Doc(
            text='bonjour, how are you today',
            tokens=[
                Token(text='bonjour', is_oov=True),
                Token(text=',', is_oov=False),
                Token(text='how', is_oov=False),
                Token(text='are', is_oov=False),
                Token(text='you', is_oov=False),
                Token(text='today', is_oov=False),
            ],
        ),
    ]

    expected = ['I am spacy, how are you']

    actual = filter_data_by_targeted_oov(target='spacy', docs=sequences)
    actual = list(actual)

    assert len(actual) == 1
    assert expected == actual

