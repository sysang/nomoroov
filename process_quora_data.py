import re
import csv

import spacy


def has_oov(doc):
    return len([token for token in doc if token.is_oov]) > 0


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')

    saved_file = f'datasets/processed-quora-duplicated-questions.csv'
    with open(saved_file, mode='w') as fwrite:
        fwrite.write('sample1\tsample2\trandom_similarity\n')

        with open('datasets/quora-duplicate-questions.tsv',  mode='r', encoding='utf-8') as fd:
            reader = csv.reader(fd, delimiter='\t')

            next(reader)

            for row in reader:
                if len(row) != 6:
                    continue
                question1 = row[3]
                question2 = row[4]
                is_duplicated = row[5]

                if is_duplicated == '0':
                    continue

                doc1 = nlp(question1)
                doc2 = nlp(question2)

                if has_oov(doc1) or has_oov(doc2):
                    continue

                data = f'{question1}\t{question2}\t{1.0}\n'
                fwrite.write(data)
