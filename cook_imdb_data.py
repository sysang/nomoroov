import csv
import re


saved_file = f'datasets/processed-imdb-movie-rating.txt'
with open(saved_file, mode='w') as fd:
    fd.write('')

with open('datasets/imdb-movie-rating.csv',  mode='r', encoding='utf-8') as fd:
    reader = csv.reader(fd)

    for row in reader:
        if len(row) != 2:
            continue
        text = row[0]
        lines = text.split('<br /><br />')
        for line in lines:
            sentences = re.split(r'\.|\?|!\s', line)
            for sent in sentences:
                sent = sent.strip()
                if not sent or len(sent.split(' ')) < 3 or len(sent.split(' ')) > 40:
                    continue
                with open(saved_file, mode='a') as fd:
                    fd.write(f'{sent}\n')
