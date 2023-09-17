from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

from typing import List, Union, Tuple


class BaseVectorizer(ABC):
    def __init__(self, path: Union[str, Path]) -> None:
        self._embedding_size: int = 0
        self.save_folder: str = ""
        pass
    
    @abstractmethod
    def fit(self, corpus: List[List[str]]) -> None:
        pass
    
    @abstractmethod
    def transform(self, text: List[str]) -> List[float]:
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        pass
    
    @abstractmethod
    def _is_exists(self, path: Path) -> bool:
        pass
    
    @abstractproperty
    def embedding_size(self) -> int:
        pass


class SparseToDenseConverter():
    def __init__(self, embedding_size: int) -> None:
        self.embedding_size = embedding_size
    
    def convert(self, sparse_vector: List[Tuple[int, float]]) -> List[float]:
        vector = [0.] * self.embedding_size
        for i, value in sparse_vector:
            vector[i] = value
        return vector