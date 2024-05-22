import msgspec


class SentencePair(msgspec.Struct):
    sample1: list[int]
    sample2: list[int]
    sim_lower_r1: float
    sim_upper_r2: float
