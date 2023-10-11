from pathlib import Path
import os
import numpy as np
from collections import defaultdict

from typing import Union, List, Dict, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PointIdsList, Record
from embedder.base import BaseVectorizer
from preprocessor.json_preparator import BaseJsonPreparator
from preprocessor.query_preparator import QueryPreparator

from utils import VectorInfoObject, InfoDocumentObject, SCROLL_LIMIT, SIMILARITY_THRESHOLD


class SingletonQdrant:
    def __new__(cls, path: str) -> QdrantClient:
        if not hasattr(cls, "instance"):
            cls.instance = QdrantClient(path=path)
        return cls.instance


class QdrantDatabase:
    def __init__(self, name: str, index: QdrantClient,  model: BaseVectorizer) -> None:
        self._index = index
        self._name = name
        self._path = os.path.join(self._index._client.location, name)
        self._model = model
        
        self._embedding_size = self._model.embedding_size
        self._query_preparator = QueryPreparator(model=self._model)
        self._json_preparator: BaseJsonPreparator
    
    
    def init_vectors(self, collection_name: str, vectors: List[VectorInfoObject]) -> None:
        self._create_collection(
            collection_name=collection_name,
            embedding_size=self._embedding_size,
        )
        self._add_vectors(collection_name=collection_name, items=vectors)

    def add_vectors(
        self,
        json_items: List[Dict],
        collection_name: Optional[str] = None,
    ) -> None:
        if not collection_name:
            collection_name = self._name

        vectors = self._get_vector_objects(json_items=json_items)
        self._add_vectors(items=vectors, collection_name=collection_name)

    def update_vectors(
        self,
        json_items: List[Dict],
        collection_name: Optional[str] = None,
    ) -> None:
        if not collection_name:
            collection_name = self._name

        vectors = self._get_vector_objects(json_items=json_items)
        self._update_vectors(vectors=vectors, collection_name=collection_name)

    def delete_vectors(
        self,
        ids: List[int],
        collection_name: Optional[str] = None,
    ) -> None:
        if not collection_name:
            collection_name = self._name
            
        if isinstance(ids, int):
            ids = [ids]
        
        ids_to_delete = [
            item.id for item in self.get_by_ids(ids, collection_name=collection_name)
        ]
            
        self._index.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=ids_to_delete),
        )
            
    def search(
        self,
        query: str,
        limit: int=5,
        collection_name: Optional[str] = None,
        ) -> str:
        if not collection_name:
            collection_name = self._name
            
        query_vector = self._query_preparator.process(query)
        search_result = self._index.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=None,
            with_vectors=False,
        )
        return search_result
    
    def search_similar(
        self,
        query: str,
        limit: int=5,
        score_threshold: float = SIMILARITY_THRESHOLD,
        collection_name: Optional[str] = None,
        ) -> str:
        if not collection_name:
            collection_name = self._name
            
        query_vector = self._query_preparator.process(query)
        search_result = self._index.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_vectors=False,
        )
        return search_result
    
    def update_index(
        self,
        collection_name: Optional[str] = None,
    ):
        if not collection_name:
            collection_name = self._name

        info_objects = self._collect_payloads(collection_name=collection_name)
        lemmatized_documents = self._lemmatize_documents(documents=info_objects)
        
        self._model.fit(corpus=lemmatized_documents)
        self._create_collection(collection_name=collection_name, embedding_size=self._model.embedding_size)
        
        items_vectors: List[VectorInfoObject] = []
        for item, lemma in zip(info_objects, lemmatized_documents):
            vector = self._model.transform(lemma)
            items_vectors.append(
                VectorInfoObject(
                    id=item.id,
                    content=item.content,
                    vector=vector,
                )
            )
        
        self._update_vectors(vectors=items_vectors, collection_name=collection_name)
        self.save()

    def find_duplicates(
        self,
        score_threshold: float = SIMILARITY_THRESHOLD,
        collection_name: Optional[str] = None,
    ):
        if not collection_name:
            collection_name = self._name
            
        vectors = self._collect_vectors(collection_name=collection_name)
        
        duplicates_dict: Dict[int, Set] = defaultdict(set)
        
        for item in vectors:
            search_result = self._index.search(
                collection_name=collection_name,
                query_vector=item.vector,
                limit=5,
                score_threshold=score_threshold,
                with_vectors=False,
            )
            if len(search_result) > 0:
                for result in search_result:
                    if item.id != result.id:
                        duplicates_dict[item.id].add(result.id)
                        duplicates_dict[item.id].add(item.id)
        res = []
        for pairs in duplicates_dict.values():
            if pairs not in res:
                res.append(pairs) 
        return res

    def get_by_ids(
        self,
        ids: List[int],
        collection_name: Optional[str] = None,
    ):
        if not collection_name:
            collection_name = self._name
        
        return self._index.retrieve(collection_name=collection_name, ids=ids)

    def save(self) -> None:
        self._model.save(self._path)
        self._json_preparator.save(self._path)
    
    @classmethod
    def load(cls, name: str, index: QdrantClient,  model: BaseVectorizer) -> None:
        model.load(os.path.join(index._client.location, name))
        json_preparator = BaseJsonPreparator.load(os.path.join(index._client.location, name))
        obj = cls(name=name, index=index, model=model)
        obj._set_json_preparator(json_preparator)
        return obj

    def _create_collection(
        self,
        collection_name: str,
        embedding_size: int,
    ) -> None:
        
        self._index.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            on_disk_payload=True
        )

    def _add_vectors(
        self,
        items: List[VectorInfoObject],
        collection_name: Optional[str] = None,
    ) -> None:
        if not collection_name:
            collection_name = self._name

        for item in items:
            self._index.upsert(
                collection_name=collection_name,
                wait=True,
                points=[
                    PointStruct(id=item.id, vector=item.vector, payload={"content": item.content}),
                ]
            )

    def _update_vectors(
        self,
        vectors: List[VectorInfoObject],
        collection_name: Optional[str] = None,
    ) -> None:
        if not collection_name:
            collection_name = self._name

        for item in vectors:
            self._index.upsert(
                    collection_name=collection_name,
                    wait=True,
                    points=[
                        PointStruct(id=item.id, vector=item.vector, payload={"content": item.content}),
                    ]
                )
            self._index.update_vectors(
                collection_name=collection_name,
                points=[
                    PointStruct(id=item.id, vector=item.vector)
                ]
            )

    def _set_json_preparator(self, preparator: BaseJsonPreparator) -> None:
        self._json_preparator = preparator
        
    def _get_vector_objects(self, json_items: List[Dict]) -> List[VectorInfoObject]:
        info_objects = self._json_preparator.convert_json(json_items)
        vectors: List[VectorInfoObject] = []
        for item in info_objects:
            document_vector = self._query_preparator.process(item.content)
            vectors.append(
                VectorInfoObject(
                    id=item.id,
                    content=item.content,
                    vector=document_vector,
                )
            )
        return vectors
    
    def _scroll_storage(
        self,
        with_payload: bool = False,
        with_vectors: bool = False,
        collection_name: Optional[str] = None,
    ) -> List[Record]:
        if not collection_name:
            collection_name = self._name

        limit = SCROLL_LIMIT
        info_objects: List[Record] = []
        next_id = 0
        offset = None
        
        while next_id is not None:
            batch, next_id = self._index.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            info_objects.extend(batch)
            offset = next_id
        return info_objects
    
    def _collect_payloads(
        self,
        collection_name: Optional[str] = None,
    ) -> List[InfoDocumentObject]:
        if not collection_name:
            collection_name = self._name
        
        payloads = self._scroll_storage(
            with_payload=True,
            with_vectors=False,
            collection_name=collection_name,
        )
        return [
            InfoDocumentObject(id=item.id, content=item.payload.get("content"))
            for item in payloads
        ]

    def _collect_vectors(
        self,
        collection_name: Optional[str] = None,
    ) -> List[VectorInfoObject]:
        if not collection_name:
            collection_name = self._name
        
        vectors = self._scroll_storage(
            with_payload=False,
            with_vectors=True,
            collection_name=collection_name,
        )
        return [
            VectorInfoObject(id=item.id, vector=item.vector, content='')
            for item in vectors
        ]

    def _lemmatize_documents(self, documents: List[InfoDocumentObject]) -> List[List[str]]:
        lemmatized_documents: List[List[str]] = []
        for document in documents:
            lemmatized_documents.append(
                self._query_preparator.lemmatize(document.content)
            )
        return lemmatized_documents


class FAQQdrantDatabase(QdrantDatabase):
    def __init__(
        self,
        name: str,
        index: QdrantClient,
        model: BaseVectorizer,
        questions_collection_name: str,
        answers_collection_name: str,
    ) -> None:
        super().__init__(name, index, model)
        self._questions_collection_name = questions_collection_name
        self._answers_collection_name = answers_collection_name
        
    def add_vectors(
        self,
        json_faq: List[Dict],
    ) -> None:
        question_vectors, answer_vectors = self._get_vector_objects(json_faq=json_faq)
        self._add_vectors(items=question_vectors, collection_name=self._questions_collection_name)
        self._add_vectors(items=answer_vectors, collection_name=self._answers_collection_name)

    def update_vectors(
        self,
        json_faq: List[Dict],
        collection_name: Optional[str] = None,
    ) -> None:
        if not collection_name:
            collection_name = self._name

        question_vectors, answer_vectors = self._get_vector_objects(json_faq=json_faq)
        self._update_vectors(vectors=question_vectors, collection_name=self._questions_collection_name)
        self._update_vectors(vectors=answer_vectors, collection_name=self._answers_collection_name)

    def delete_vectors(
        self,
        ids: List[int],
        collection_name: Optional[str] = None,
    ) -> None:
        if not collection_name:
            collection_name = self._name
            
        if isinstance(ids, int):
            ids = [ids]
            
        ids_to_delete = [
            item.id for item in self.get_by_ids(ids, collection_name=self._questions_collection_name)
        ]
            
        self._index.delete(
            collection_name=self._questions_collection_name,
            points_selector=PointIdsList(points=ids_to_delete),
        )
        self._index.delete(
            collection_name=self._answers_collection_name,
            points_selector=PointIdsList(points=ids_to_delete),
        )
            
    def search(
        self,
        query: str,
        limit: int=5,
        ) -> str:
        query_vector = self._query_preparator.process(query)
        search_questions = self._index.search(
            collection_name=self._questions_collection_name,
            query_vector=query_vector,
            limit=limit,
            with_vectors=False,
        )
        search_answers = self._index.search(
            collection_name=self._answers_collection_name,
            query_vector=query_vector,
            limit=limit,
            with_vectors=False,
        )
        result = sorted(list(search_questions+search_answers), key=lambda x: -x.score)

        result_ids: list[int] = []
        for item in result:
            if item.id not in result_ids:
                result_ids.append(item.id)

        return self.get_by_ids(ids=result_ids[:limit], collection_name=self._answers_collection_name)

    def search_similar(
        self,
        query: str,
        limit: int=5,
        score_threshold: float = SIMILARITY_THRESHOLD,
        ) -> str:  
        query_vector = self._query_preparator.process(query)
        search_result = self._index.search(
            collection_name=self._answers_collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_vectors=False,
        )
        return search_result
    
    def update_index(self):
        questions_objects = self._collect_payloads(collection_name=self._questions_collection_name)
        answers_objects = self._collect_payloads(collection_name=self._answers_collection_name)
        lemmatized_questions = self._lemmatize_documents(documents=questions_objects)
        lemmatized_answers = self._lemmatize_documents(documents=answers_objects)
        
        self._model.fit(corpus=lemmatized_questions + lemmatized_answers)
        self._create_collection(collection_name=self._questions_collection_name, embedding_size=self._model.embedding_size)
        self._create_collection(collection_name=self._answers_collection_name, embedding_size=self._model.embedding_size)
        
        questions_vectors: List[VectorInfoObject] = []
        answers_vectors: List[VectorInfoObject] = []
        for item, lemma in zip(questions_objects, lemmatized_questions):
            vector = self._model.transform(lemma)
            questions_vectors.append(
                VectorInfoObject(
                    id=item.id,
                    content=item.content,
                    vector=vector,
                )
            )
        for item, lemma in zip(answers_objects, lemmatized_answers):
            vector = self._model.transform(lemma)
            answers_vectors.append(
                VectorInfoObject(
                    id=item.id,
                    content=item.content,
                    vector=vector,
                )
            )
        
        self._update_vectors(vectors=questions_vectors, collection_name=self._questions_collection_name)
        self._update_vectors(vectors=answers_vectors, collection_name=self._answers_collection_name)
        self.save()

    def find_duplicates(
        self,
        score_threshold: float = SIMILARITY_THRESHOLD,
    ):
        vectors = self._collect_vectors(collection_name=self._answers_collection_name)
        
        duplicates_dict: Dict[int, Set] = defaultdict(set)
        
        for item in vectors:
            search_result = self._index.search(
                collection_name=self._answers_collection_name,
                query_vector=item.vector,
                limit=5,
                score_threshold=score_threshold,
                with_vectors=False,
            )
            if len(search_result) > 0:
                for result in search_result:
                    if item.id != result.id:
                        duplicates_dict[item.id].add(result.id)
                        duplicates_dict[item.id].add(item.id)
        res = []
        for pairs in duplicates_dict.values():
            if pairs not in res:
                res.append(pairs) 
        return res

    @classmethod
    def load(
        cls,
        name: str,
        index: QdrantClient,
        model: BaseVectorizer,
        questions_collection_name: str,
        answers_collection_name: str,
    ) -> None:
        model.load(os.path.join(index._client.location, name))
        json_preparator = BaseJsonPreparator.load(os.path.join(index._client.location, name))
        obj = cls(
            name=name,
            index=index,
            model=model,
            questions_collection_name=questions_collection_name,
            answers_collection_name=answers_collection_name,
        )
        obj._set_json_preparator(json_preparator)
        return obj

    def _get_vector_objects(self, json_faq: List[Dict]) -> List[VectorInfoObject]:
        info_objects = self._json_preparator.convert_json(json_faq)
        question_vectors: List[VectorInfoObject] = []
        answer_vectors: List[VectorInfoObject] = []
        for item in info_objects:
            question_vector = self._query_preparator.process(item.question)
            answer_vector = self._query_preparator.process(item.answer)
            question_vectors.append(
                VectorInfoObject(
                    id=item.id,
                    content=item.question,
                    vector=question_vector,
                )
            )
            answer_vectors.append(
                VectorInfoObject(
                    id=item.id,
                    content=item.answer,
                    vector=answer_vector,
                )
            )
        return question_vectors, answer_vectors
