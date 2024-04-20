import hashlib
import itertools
from typing import Iterable

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


def cook_training_data(list_of_num, start=0):
    for num in list_of_num:
        it0 = itertools.tee(list_of_num, 1)[0]
        cook_training_data(it0)


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
