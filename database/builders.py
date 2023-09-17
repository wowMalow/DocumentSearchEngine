from pathlib import Path

from utils import InfoObject, InfoDocumentObject, VectorInfoObject, LemmaInfoObject
from preprocessor.lemmatizer import Lemmatizer
from preprocessor.json_preparator import BaseJsonPreparator, JsonPreparator, FAQJsonPreparator
from embedder.base import BaseVectorizer
from database.qdrant import QdrantDatabase, FAQQdrantDatabase

from typing import Union, List, Dict, Tuple


class QdrantDatabaseBuilder:
    def __init__(self, index: QdrantDatabase) -> None:
        self._lemmatizer = Lemmatizer()
        self._index = index

    def build_database(
        self,
        name: str,
        json_items: List[Dict],
        id_field: str,
        content_field: str,
        collection_name: str,
        model: BaseVectorizer,
    ) -> QdrantDatabase:
        json_preparator = JsonPreparator(id_field=id_field, content_field=content_field)
        info_objects = json_preparator.convert_json(json_items)
        lemmatized_documents = self._lemmatize_documents(info_objects)
        
        model = self._train_model(
            model=model,
            lemmatized_documents=lemmatized_documents,
        )
        
        documents_vectors = self._prepare_vectors(
            lemmatized_items=lemmatized_documents,
            model=model,
        )

        self.database = QdrantDatabase(name=name, index=self._index, model=model)
        self.database.init_vectors(
            collection_name=collection_name,
            vectors=documents_vectors,
        )
        self.database._set_json_preparator(preparator=json_preparator)
        self.database.save()
        return self.database

    def build_faq_database(
        self,
        name: str,
        faq_json: List[Dict],
        id_field: str,
        question_field: str,
        answer_field: str,
        questions_collection_name: str,
        answers_collection_name: str,
        model: BaseVectorizer,
    ) -> FAQQdrantDatabase:
        json_preparator = FAQJsonPreparator(
            id_field=id_field,
            question_field=question_field,
            answer_field=answer_field,
        )
        info_objects = json_preparator.convert_json(faq_json)
           
        lemmatized_questions, lemmatized_answers = self._lemmatize_faq(info_objects)
        
        model = self._train_model(
            model=model,
            lemmatized_documents=lemmatized_questions + lemmatized_answers,
        )
        
        questions_vectors = self._prepare_vectors(
            lemmatized_items=lemmatized_questions,
            model=model,
        )
        answers_vectors = self._prepare_vectors(
            lemmatized_items=lemmatized_answers,
            model=model,
        )

        self.database = FAQQdrantDatabase(
            name=name,
            index=self._index,
            model=model,
            questions_collection_name=questions_collection_name,
            answers_collection_name=answers_collection_name,
        )
        self.database.init_vectors(collection_name=questions_collection_name, vectors=questions_vectors)
        self.database.init_vectors(collection_name=answers_collection_name, vectors=answers_vectors)
        self.database._set_json_preparator(preparator=json_preparator)
        self.database.save()
        return self.database
    
    def _lemmatize_faq(self, info_objects: List[InfoObject]) -> Tuple[List[LemmaInfoObject]]:
        lemmatized_questions: List[LemmaInfoObject] = []
        lemmatized_answers: List[LemmaInfoObject] = []
        for item in info_objects:
            lemmatized_question = self._lemmatizer.process(text=item.question)
            lemmatized_answer = self._lemmatizer.process(text=item.answer)
            lemmatized_questions.append(
                LemmaInfoObject(
                    id=item.id,
                    content=item.question,
                    lemmas=lemmatized_question
                )
            )
            lemmatized_answers.append(
                LemmaInfoObject(
                    id=item.id,
                    content=item.answer,
                    lemmas=lemmatized_answer
                )
            )
        return lemmatized_questions, lemmatized_answers

    def _lemmatize_documents(self, info_objects: List[InfoDocumentObject]) -> List[LemmaInfoObject]:
        lemmatized_documents: List[LemmaInfoObject] = []
        for item in info_objects:
            lemmatized_document = self._lemmatizer.process(text=item.content)
            lemmatized_documents.append(
                LemmaInfoObject(
                    id=item.id,
                    content=item.content,
                    lemmas=lemmatized_document
                )
            )
        return lemmatized_documents
    
    def _get_corpus(
        self,
        lemmatized_documents: List[LemmaInfoObject],
    ) -> List[List[str]]:
        lemmas_corpus: List[List[str]] = []
        for item in lemmatized_documents:
            lemmas_corpus.append(item.lemmas)
        return lemmas_corpus

    def _train_model(
        self,
        model: BaseVectorizer,
        lemmatized_documents: List[LemmaInfoObject],
    ) -> BaseVectorizer:
        lemmas_corpus = self._get_corpus(lemmatized_documents)
        model.fit(corpus=lemmas_corpus)
        return model
        
    def _prepare_vectors(
        self,
        lemmatized_items: List[LemmaInfoObject],
        model: BaseVectorizer,
    ) -> List[VectorInfoObject]:
        item_vectors: List[VectorInfoObject] = []
        for item in lemmatized_items:
            vector = model.transform(item.lemmas)
            item_vectors.append(
                VectorInfoObject(
                    id=item.id,
                    content=item.content,
                    vector=vector,
                )
            )
        return item_vectors

