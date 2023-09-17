import os
from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
import pickle
from typing import List, Tuple

from utils import  InfoObject, InfoDocumentObject


class BaseJsonPreparator(ABC):
    @abstractmethod
    def convert_json(self, json: List[dict]) -> List:
        pass
    
    @abstractstaticmethod
    def filter_updated(
        new_items: List,
        old_items: List,
    ) -> Tuple[List]:
        pass

    def save(self, path: str) -> None:
        with open(os.path.join(path, 'json_preparator.bin'), 'wb') as file:
            pickle.dump(self, file) 
    
    @classmethod
    def load(cls, path: str) -> "BaseJsonPreparator":
        with open(os.path.join(path, 'json_preparator.bin'), 'rb') as file:
            preparator_dict = pickle.load(file)
        return preparator_dict


class JsonPreparator(BaseJsonPreparator):
    def __init__(self, id_field: str, content_field: str) -> None:
        self._id = id_field
        self._content = content_field
    
    def convert_json(self, json: List[dict]) -> List[InfoDocumentObject]:
        filtered = []
        for item in json:
            item_id = item.get(self._id)
            content = item.get(self._content)
            if content and item_id and len(content) > 1:
                filtered.append(
                    InfoDocumentObject(
                        id=item_id,
                        content=content,
                    )
                )
        return filtered
    
    @staticmethod
    def filter_updated(
        new_items: List[InfoDocumentObject],
        old_items: List[InfoDocumentObject],
    ) -> Tuple[List[InfoDocumentObject]]:
        new_items_ids = [item.id for item in new_items]
        old_items_dict = {item.id: item for item in old_items}
        
        to_update = []
        to_delete = []
        for new_item in new_items:
            old_item = old_items_dict.get(new_item.id)
            if new_item != old_item:
                to_update.append(new_item) 
    
        for old_item in old_items:
            if old_item.id not in new_items_ids:
                to_delete.append(old_item)
                
        return to_update, to_delete
                
                
class FAQJsonPreparator(BaseJsonPreparator):
    def __init__(self, id_field: str, question_field: str, answer_field: str) -> None:
        self._id = id_field
        self._question = question_field
        self._answer = answer_field
    
    def convert_json(self, json: List[dict]) -> List[InfoObject]:
        filtered = []
        for item in json:
            item_id = item.get(self._id)
            question = item.get(self._question)
            answer = item.get(self._answer)
            if question and answer and item_id and len(question) > 1 and len(answer) > 1:
                filtered.append(
                    InfoObject(
                        id=item_id,
                        question=question,
                        answer=answer,
                    )
                )
        return filtered
    
    @staticmethod
    def filter_updated(new_items: List[InfoObject], old_items: List[InfoObject]) -> Tuple[List[InfoObject]]:
        new_items_ids = [item.id for item in new_items]
        old_items_dict = {item.id: item for item in old_items}
        
        to_update = []
        to_delete = []
        for new_item in new_items:
            old_item = old_items_dict.get(new_item.id)
            if new_item != old_item:
                to_update.append(new_item) 
    
        for old_item in old_items:
            if old_item.id not in new_items_ids:
                to_delete.append(old_item)
                
        return to_update, to_delete
