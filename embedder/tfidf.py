import os
from typing import List, Union
from gensim import corpora, models

from embedder.base import BaseVectorizer, SparseToDenseConverter

class TfIdf(BaseVectorizer):
    def __init__(self) -> None:
        self.save_folder = "tfidf"
    
    def fit(self, corpus: List[List[str]]) -> None:
        self._dictionary = corpora.Dictionary(corpus)
        self._embedding_size = len(self._dictionary.token2id)
        self._converter = SparseToDenseConverter(embedding_size=self._embedding_size)

        corpus_vectorized = [self._dictionary.doc2bow(text) for text in corpus]
        self._model = models.TfidfModel(corpus_vectorized)
    
    def transform(self, text: List[str]) -> List[float]:
        kw_vector = self._dictionary.doc2bow(text)
        tfidf_vector = self._model[kw_vector]
        return self._converter.convert(sparse_vector=tfidf_vector)
    
    def save(self, path: str) -> None:
        if not os.path.exists(os.path.join(path, self.save_folder)):
            os.makedirs(os.path.join(path, self.save_folder))
        
        self._dictionary.save(os.path.join(path, self.save_folder, "dictionary.bin"))
        self._model.save(os.path.join(path, self.save_folder, "model.bin"))
    
    def load(self, path: str) -> None:
        self._dictionary = corpora.Dictionary.load(os.path.join(path, self.save_folder, "dictionary.bin"))
        self._embedding_size = len(self._dictionary.token2id)
        self._converter = SparseToDenseConverter(embedding_size=self._embedding_size)
        self._model = models.TfidfModel.load(os.path.join(path, self.save_folder, "model.bin"))
    
    @property
    def embedding_size(self):
        return self._embedding_size
    
    def _is_exists(self, path: str) -> bool:
        pass