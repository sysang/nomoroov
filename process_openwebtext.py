from openwebtext import Openwebtext


builder = Openwebtext()
builder.download_and_prepare(download_mode='reuse_dataset_if_exists')
dataset = builder.as_dataset(split='train')

saved_file = 'datasets/urlsf_subset00_{:02d}.txt'

batch_size = 300000
batch = []
batch_count = 0
batch_num = 0

for sample in dataset:
    text = sample['text']
    lines = text.split('\n\n')
    for line_ in lines:
        line = line_.strip()
        if len(line.split()) < 5 or len(line.split()) > 40:
            continue
        batch.append(line)
        batch_count += 1

        if batch_count > batch_size:
            with open(saved_file.format(batch_num), mode='w') as fw:
                fw.write('\n'.join(batch))
                batch = []
                batch_count = 0
                batch_num += 1
