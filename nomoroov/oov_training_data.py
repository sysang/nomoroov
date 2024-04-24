import hashlib
import itertools
import abc
from typing import Iterable, Any, Union

from .base_nlp import Doc, Token
from .get_logger import get_logger


logger = get_logger(__file__)


TRAINING_DATA_DIR = 'oov_training_data'


class DataCollectorBase(abc.ABC):
    @abc.abstractmethod
    def collect(self, item: Any):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass


class PseudoDataCollector(DataCollectorBase):
    def collect(self, item: Any):
        print(item)

    def finalize(self):
        pass


class InMemoryDataCollector(DataCollectorBase):
    def __init__(self):
        self._data = []

    def collect(self, item):
        self._data.append(item)

    @property
    def data(self):
        return self._data

    def finalize(self):
        pass


class CsvFileDataCollector(DataCollectorBase):
    def __init__(self, filepath, batch_size=500):
        self.filepath = filepath
        self.batch_size = batch_size

        self.init_state()

        # Clear data if file exists
        with open(filepath, mode='w') as fd:
            logger.info(f'Clear all data in  file {filepath} (if exists)')
            fd.write('')

    def init_state(self):
        self.data = []
        self.counter = 0
        self.batch_number = 1

    def collect(self, item: tuple[Any, Any]):
        self.data.append(item)
        self.counter += 1

        if self.counter >= self.batch_size:
            self.save_data()
            self.finalize()

    def finalize(self):
        self.save_data()
        self.init_state()

    def save_data(self):
        with open(self.filepath, mode='a') as fd:
            for item in self.data:
                item0, item1 = (item[0].strip(), item[1].strip())
                fd.write(f"{item0}\t{item1}\n")
            logger.info(f'Store data to {self.filepath}, batch size:\
                    {self.batch_size}, batch number: {self.batch_number}')


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
        if encoded in caches:
            return True
        else:
            add_to_cache(text)
            return False


    return (is_cached, add_to_cache)


def is_short_doc(doc: Doc):
    return len(doc.tokens) < 6;


def has_one_oov(doc: Doc) -> Union[Token, None]:
    oov_count = 0
    oov_token = None
    for token in doc.tokens:
        if token.is_oov:
            oov_count += 1
            oov_token = token

        if oov_count >= 2: 
            return None

    if oov_count == 0:
        return None

    return oov_token


def filter_data_by_targeted_oov(target: str, docs: Iterable[Doc]) -> Iterable[str]:
    """ Include only text that has one oov, which is the target. """

    is_cached, _ = create_cache()

    for doc in docs:
        # To exclude duplicated texts
        if is_cached(doc.text):
            continue

        # To exclude short text
        if is_short_doc(doc):
            continue

        oov_token = has_one_oov(doc)
        if oov_token is not None and oov_token.text == target:
            yield doc.text


def filter_data_by_unrecognized_oov(targets: str, unrecognized_token: str, docs: Iterable[Doc]):
    """ 
        Include only text that has one oov, which is the unrecognized token
        (one that is not in list of targeted tokens)
    """

    is_cached, _ = create_cache()

    for doc in docs:
        # To exclude duplicated texts
        if is_cached(doc.text):
            continue

        # To exclude short text
        if is_short_doc(doc):
            continue

        oov_token = has_one_oov(doc)
        if oov_token is not None and oov_token.text not in targets:
            yield doc.text.replace(oov_token.text, unrecognized_token)


def _cook_training_data(
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
            ended_level, end_item, current_counter = _cook_training_data(
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
            ended_level, end_item, counter = _cook_training_data(
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


def cook_training_data(
    list_of_item: Iterable[Any],
    data_collector: DataCollectorBase = PseudoDataCollector(),
    window_size=50
):
    """ Wrapper to ensure that data_collector always store remaining data (not stacked to one full batch)"""
    _cook_training_data(
        list_of_item=list_of_item,
        data_collector=data_collector,
        window_size=window_size,
    )
    data_collector.finalize();
