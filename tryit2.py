import torch

from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG
from load_spacy import load_spacy


CFG['dropout1'] = 0.0
CFG['dropout2'] = 0.0
CFG['batch_size'] = 1
CFG['device'] = 'cpu'

checkpoint = 'tmp/checkpoints/v3/epoch2_encoder1'
# checkpoint = 'tmp/checkpoints/batches/v3/epoch1_batch2400_encoder2'


model = SentenceEmbedding(CFG).to('cpu')
model.load_state_dict(torch.load(checkpoint))
model.eval()

print('number of parameters: ', sum( p.numel() for p in model.parameters() if p.requires_grad))

nlp = load_spacy()

tests = [
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "How can I be a good geologist?",
    ),
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "How do I read and find my YouTube comments?",
    ),
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "What can make Physics easy to learn?",
    ),
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "What was your first sexual experience like?",
    ),
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "What would a Trump presidency mean for current international master’s students on an F1 visa?",
    ),
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "What does manipulation mean?",
    ),
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "Why are so many Quora users posting questions that are readily answered on Google?",
    ),
    (
        "How can I be a good geologist?",
        "How do I read and find my YouTube comments?",
    ),
    (
        "How can I be a good geologist?",
        "What can make Physics easy to learn?",
    ),
    (
        "How can I be a good geologist?",
        "What was your first sexual experience like?",
    ),
    (
        "How can I be a good geologist?",
        "What would a Trump presidency mean for current international master’s students on an F1 visa?",
    ),
    (
        "How can I be a good geologist?",
        "What does manipulation mean?",
    ),
    (
        "How can I be a good geologist?",
        "Why are so many Quora users posting questions that are readily answered on Google?",
    ),
    (
        "Another groundhog year of Brexit ushers in a decade of disruption",
        "How do I read and find my YouTube comments?",
    ),
    (
        "Another groundhog year of Brexit ushers in a decade of disruption",
        "What can make Physics easy to learn?",
    ),
    (
        "How can I be a good geologist?",
        "What was your first sexual experience like?",
    ),
    (
        "Another groundhog year of Brexit ushers in a decade of disruption",
        "What would a Trump presidency mean for current international master’s students on an F1 visa?",
    ),
    (
        "Another groundhog year of Brexit ushers in a decade of disruption",
        "What does manipulation mean?",
    ),
    (
        "Another groundhog year of Brexit ushers in a decade of disruption",
        "Why are so many Quora users posting questions that are readily answered on Google?",
    ),
    (
        "Another groundhog year of Brexit ushers in a decade of disruption",
        "In the coronavirus jobs wipeout",
    ),
]

for text1, text2 in tests:
    score = model.similarity(text1, text2, nlp)

    print(f'{score:5f}\t"{text1}" "{text2}"')
