import os
import hashlib


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


def process_data():
    is_cached, add_to_cache = create_cache()

    for file in os.listdir('datasets/books'):
        if file.endswith(".txt"):
            with open(f'datasets/books/{file}', mode='r', encoding='utf-8') as fd:
                content = fd.read()
                paragraphs = content.split('\n\n')
                for block in paragraphs:
                    block = block.strip()

                    if not block: 
                        continue

                    if not block.endswith('.'):
                        continue

                    paragraph = block.split('.\n')
                    for splitted_lines in paragraph:
                        lines = ' '.join([line_.strip() for line_ in splitted_lines.strip().splitlines()])

                        for line_ in lines.split('. '):
                            line = line_.strip()

                            if len(line) < 11:
                                continue
                            
                            if is_cached(line):
                                continue

                            if len(line) > 200:
                                continue

                            add_to_cache(line)
                            print(line)


if __name__ == '__main__':
    process_data()
