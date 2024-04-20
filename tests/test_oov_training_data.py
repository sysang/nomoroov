import itertools

from nomoroov.oov_training_data import cook_training_data


def test_cook_training_data():
    list_of_num = iter(range(1, 11))
    cook_training_data(list_of_num)
    assert True
