import os
import csv

from peewee import SqliteDatabase, Model, BlobField
import msgspec

from data_schema import SentencePair


class BaseModel(Model):
    json_data = BlobField()


def test_data(dbname):
    db = SqliteDatabase(f'sentence_embedding_training_data/{dbname}.db')

    class Record(BaseModel):
        table_name = 'record'
        class Meta:
            database = db

    db.connect()

    counter = 0
    for record in Record.select():
        counter += 1
        if counter % 1000 == 0:
            print(msgspec.json.decode(record.json_data, type=SentencePair))

    db.close()


def get_model_class(db):
    class Record(BaseModel):
        table_name = 'record'
        class Meta:
            database = db
    
    return Record


def merge_databases():
    datasets = [
            # 'wikidata-text-part-0',
            # 'wikidata-text-part-1',
            # 'wikidata-text-part-2',
            # 'wikidata-text-part-3',
            # 'wikidata-text-part-4',
            # 'wikidata-text-part-5',
            # 'abcnews-date-text',
            # 'processed-imdb-movie-rating',
            # 'cnbc_headlines',
            # 'reuters_headlines',
            # 'guardian_headlines',
            'gutenberg-project-book-part-0',
            'gutenberg-project-book-part-1',
            'gutenberg-project-book-part-2',
    ]

    dbname = 'sqlite_file'
    dbfile = f'sentence_embedding_training_data/{dbname}.db'

    # if os.path.exists(dbfile):
    #     print(f'[INFO] Removed database {dbfile}.')
    #     os.remove(dbfile)

    merged_db = SqliteDatabase(dbfile)

    class Record(BaseModel):
        table_name = 'record'
        class Meta:
            database = merged_db

    merged_db.connect()
    # merged_db.create_tables([Record])

    total_quantity = 0

    for dataset in datasets:
        batch = []
        counter = 0

        db = SqliteDatabase(f'sentence_embedding_training_data/{dataset}.txt.db')

        ModelClass = get_model_class(db)

        db.connect()

        for record in ModelClass.select():
            batch.append({'json_data': record.json_data})
            counter += 1
            total_quantity += 1

            if counter >= 10000:
                Record.insert_many(batch).execute()
                batch = []
                counter = 0

        if len(batch) > 0:
            Record.insert_many(batch).execute()
            print(f'Merged dataset: {dataset}, total quantity: {total_quantity}.')


        db.close()

    merged_db.close()


if __name__ == '__main__':
    # merge_databases()

    dbname = 'gutenberg-project-book-part-0'
    test_data(dbname)

