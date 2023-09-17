from pathlib import Path
from typing import List, Union

from embedder.base import BaseVectorizer

class Doc2Vector(BaseVectorizer):
    def __init__(self, path: Union[str, Path]) -> None:
        self._path = path
        if self._is_exists(self._path):
            self._bag = self.load()
    
    def fit(self, corpus: List[List[str]]) -> None:
        pass
    
    def transform(self, text: List[str]) -> List[float]:
        pass
    
    def save(self) -> None:
        pass
    
    def load(self) -> None:
        pass
    
    def _is_exists(self, path: Union[str, Path]) -> bool:
        pass

