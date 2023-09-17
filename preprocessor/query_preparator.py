from preprocessor.lemmatizer import Lemmatizer

from typing import List


class QueryPreparator:
    def __init__(self, model) -> None:
        self._lemmatizer = Lemmatizer()
        self._model = model

    def process(self, text: str) -> List[float]:
        lemmatized_text = self._lemmatizer.process(text=text)
        return self._model.transform(lemmatized_text)
    
    def lemmatize(self, text: str) -> List[str]:
        return self._lemmatizer.process(text=text)
        