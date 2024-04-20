import spacy

from .base_nlp import BaseNlp, Token, Doc


class SpacyNlp(BaseNlp):
    def __init__(self, model_path: str):
        self.nlp = spacy.load(model_path)

    def tokenize(self, text: str) -> Doc:
        doc = self.nlp(text)
        # IMPORTANT: invoke strip() to git rid of "\n'
        tokens = [Token(text=tok.text, is_oov=tok.is_oov) for tok in doc if len(tok.text.strip()) > 0]

        return Doc(tokens=tokens, text=text)
