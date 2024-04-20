from nomoroov.spacy_nlp import SpacyNlp
from nomoroov.datasource import TextFileDatasource
from nomoroov.analyse_oov_in_corpus import analyse_oov_in_corpus


nlp = SpacyNlp('en_core_web_md')
datasource = TextFileDatasource('datasets')

data = analyse_oov_in_corpus(nlp, datasource)
print(data)
