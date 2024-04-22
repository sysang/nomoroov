import hashlib
import itertools
import abc
from typing import Iterable, Any

from .base_nlp import Doc


# def generate(target):
#     caches = set()
#     raw_data = []
#     for file in corpuses:
#         with open(f"corpus/{file}", newline='') as fd:
#             lines = fd.readlines()
#             for line_ in list (lines):
#                 line = line_.strip()

#                 m = hashlib.sha256()
#                 m.update(line.encode())
#                 encoded = m.hexdigest()
#                 if encoded in caches:
#                     continue
#                 else:
#                     caches.add(encoded)

#                 doc = nlp(line)

#                 if len(doc) < 6:
#                     continue

#                 tokens = [ token.text for token in doc ]
#                 if target not in tokens and target != unrecognized_token:
#                     continue

#                 oov_count = 0
#                 oov_token = None

#                 for token in doc:
#                     if token.is_oov:
#                         oov_count += 1
#                         oov_token = token

#                     if oov_count >= 2: 
#                         continue

#                     if target is unrecognized_token:
#                         text = line.replace(token.text, unrecognized_token)
#                     else:
#                         text = line

#                     raw_data.append(text)

#     return save_data(target, raw_data)

TRAINING_DATA_DIR = 'oov_training_data'


class DataCollectorBase(abc.ABC):
    @abc.abstractmethod
    def collect(self, item: Any):
        pass


class PseudoDataCollector(DataCollectorBase):
    def collect(self, item: Any):
        print(item)


class InMemoryDataCollector(DataCollectorBase):
    def __init__(self):
        self._data = []

    def collect(self, item):
        self._data.append(item)

    @property
    def data(self):
        return self._data


def create_cache():
    caches = set()

    def hash_text(text: str):
        m = hashlib.sha256()
        m.update(text.encode())
        return m.hexdigest()

    def add_to_cache(text: str):
        encoded = hash_text(text)
        caches.add(encoded)

    def is_cached(text: str):
        encoded = hash_text(text)
        return encoded in caches

    return (add_to_cache, is_cached)


def filter_data_by_targeted_oov(target: str, docs: Iterable[Doc]):
    """ Include only text that has one oov, which is the target. """

    add_to_cache, is_cached = create_cache()

    for doc in docs:
        # To exclude duplicated texts
        if is_cached(doc.text):
            continue
        else:
            add_to_cache(doc.text)

        # To exclude short text
        if len(doc.tokens) < 6:
            continue

        oov_count = 0
        oov_token = None
        for token in doc.tokens:
            if token.is_oov:
                oov_count += 1
                oov_token = token

            if oov_count >= 2: 
                break
        
        if oov_count == 1 and oov_token is not None and oov_token.text == target:
            yield doc


def cook_training_data(
    list_of_item: Iterable[Any],
    data_collector: DataCollectorBase = PseudoDataCollector(),
    window_size=50,
    depth_level=0,
    ended_level=0,
    prev_item=None,
    end_item=None,
    counter=0,
):
    current_counter = counter
    for item in list_of_item:
        current_level = depth_level + 1

        if prev_item is None:
            current_counter = current_counter + 1

        if prev_item is not None:
            data_collector.collect((prev_item, item))

        if item == end_item:
            return ended_level, end_item, current_counter

        if current_level < window_size and current_level > ended_level:
            it0, it1 = itertools.tee(list_of_item)
            ended_level, end_item, current_counter = cook_training_data(
                    list_of_item=it1,
                    data_collector=data_collector,
                    window_size=window_size,
                    depth_level=current_level,
                    ended_level=ended_level,
                    prev_item=item,
                    end_item=end_item,
                    counter=current_counter + 1,
                    )

        if current_level == window_size:
            return current_level, item, current_counter

        if current_counter % window_size == 1 and prev_item is None:
            it0, it1 = itertools.tee(list_of_item)
            it2 = itertools.chain(iter([item]), it1)
            ended_level, end_item, counter = cook_training_data(
                list_of_item=it2,
                data_collector=data_collector,
                window_size=window_size,
                depth_level=0,
                ended_level=0,
                prev_item=None,
                end_item=None,
                counter=current_counter - 1,
            )

    return depth_level, end_item, current_counter


# def cook_training_data(target, docs: Iterable[doc]):
#     saved_file = f'oov_training_data/{target}_in_pairs.csv'
#     quantity = len(raw_data)

#     with open(saved_file, mode='w') as fd:

#         if target is not unrecognized_token: 
#             for i in range(quantity - 2): 
#                 for j in range(i + 1, quantity - 1):
#                      data = f"{raw_data[i]}\t{raw_data[j]}\n" 
#                      fd.write(data)
#         else:
#             caches = set() 
#             for i in range(quantity): 
#                 while True: 
#                     j = random.choice (range(quantity))
#                     if i != j and (i, j) not in caches:
#                         break
#                 caches.add((i, j)) 
#                 data = f"{raw_data[i]}\t{raw_data[j]}\n"
#                 fd.write(data)

#     return saved_file


# def save_training_data():
#     saved_file = f'{TRAINING_DATA_DIR}/{target}_in_pairs.csv'
#     with open(saved_file, mode='w') as fd:
#         pass
