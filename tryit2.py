from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG
from load_spacy import load_spacy

import torch


CFG['dropout1'] = 0.0
CFG['dropout2'] = 0.0
CFG['batch_size'] = 1
CFG['device'] = 'cpu'

# checkpoint = 'tmp/checkpoints/v2/epoch1_encoder1'
checkpoint = 'tmp/checkpoints/batches/v2/epoch1_batch2800_encoder1'


model = SentenceEmbedding(CFG).to('cpu')
model.load_state_dict(torch.load(checkpoint))
model.eval()

print('number of parameters: ', sum( p.numel() for p in model.parameters() if p.requires_grad))

nlp = load_spacy()

tests = [
    (
        "Another groundhog year of Brexit ushers in a decade of disruption",
        "In the coronavirus jobs wipeout",
    ),
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
        "What does manipulation mean? - What does manipulation means?",
    ),
    (
        "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? ",
        "Why are so many Quora users posting questions that are readily answered on Google?",
    ),

]

for text1, text2 in tests:
    score = model.similarity(text1, text2, nlp)

    print(f'{text1} - {text2} - {score}')
