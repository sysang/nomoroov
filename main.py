from nomoroov.spacy_nlp import SpacyNlp
from nomoroov.datasource import TextFileDatasource
from nomoroov.analyse_oov_in_corpus import count_oov
from nomoroov.oov_training_data import (
    filter_data_by_targeted_oov,
    cook_training_data,
    CsvFileDataCollector
)


nlp = SpacyNlp('en_core_web_md')
datasource = TextFileDatasource('datasets', nlp)
docs = datasource.read()

# data = count_oov(docs)
# print(data)

target = 'StanChart'
storing_training_data_file = f'oov_training_data/{target}_in_pair.txt'

data_collector = CsvFileDataCollector(storing_training_data_file)
list_of_item = filter_data_by_targeted_oov(target=target, docs=docs)
cook_training_data(list_of_item=list_of_item, data_collector=data_collector)



