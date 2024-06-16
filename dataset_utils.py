import math

import msgspec

from data_schema import SentencePair


def dataset_map_fn(adjust_r1=0.0, adjust_r2=0.0, adjust_threshold_by_r1=-math.inf):
    def dataset_map(samples):
        sample1 = []
        sample2 = []
        r1 = []
        r2 = []

        for sample in samples['json_data']:
            pair = msgspec.json.decode(sample, type=SentencePair)
            sample1.append(pair.sample1)
            sample2.append(pair.sample2)

            if (pair.sim_lower_r1 > adjust_threshold_by_r1):
                r1.append(pair.sim_lower_r1)
                r2.append(pair.sim_upper_r2)
            else:
                r1.append(pair.sim_lower_r1 + adjust_r1)
                r2.append(pair.sim_upper_r2 - adjust_r2)

        return {
            'sample1': sample1,
            'sample2': sample2,
            'sim_lower_r1': r1,
            'sim_upper_r2': r2,
        }

    return dataset_map


dataset_map = dataset_map_fn()

