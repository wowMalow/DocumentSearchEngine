import re
import pymorphy2
from preprocessor.tokenizer import Tokenizer

from typing import List, NamedTuple, Optional


class Lemmatizer(Tokenizer):
    def __init__(self) -> None:
        self.analizer = pymorphy2.MorphAnalyzer()
        self.stopwords = self._load_stopwords()
        
    def process(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        cleaned_tokens = self._remove_stopwords(tokens=tokens)
        lemmatized = []
        for word in cleaned_tokens:
            norm_word = self.analizer.normal_forms(word)[0]
            lemmatized.append(norm_word)
        return lemmatized
    
    def _load_stopwords(self):
        with open("russian", "r", encoding="utf-8") as file:
            stopwords = file.read().split('\n')[:-1]
        return stopwords
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords]
