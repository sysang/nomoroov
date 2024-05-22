import re
import csv


def split_corpus():
    with open('datasets/plain-text-wikipedia-simple-english.txt', mode='r', encoding='utf-8') as fd:
        paragraphs = fd.readlines()
        counter = 0
        batch_number = 0
        batch = []
        
        for text in paragraphs:
            paragraph = text.strip()
            if len(text.split(' ')) < 10:
                continue

            batch.append(paragraph)
            counter += 1

            if counter >= 100000:
                saved_file = f'datasets/wikidata-text-part-{batch_number}.txt'
                data = []
                with open(saved_file, mode='w', encoding='utf-8') as fwrite:
                    for paragraph in batch:
                        sentences = re.split(r'\.|\?|!\s', paragraph)
                        for sent in sentences:
                            sent = sent.strip()
                            if (not sent
                                or len(sent.split(' ')) < 10
                                or len(sent.split(' ')) > 40
                                or sent[-1] == ':'
                                or sent[0] == ','
                                ):
                                continue
                            data.append(sent)

                    fwrite.write('\n'.join(data))

                counter = 0
                batch = []
                batch_number += 1


if __name__ == '__main__':
    split_corpus()

