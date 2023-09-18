# Documents search engine

Module for building vector-based documents search

## Installation

```bash
pip install -r requirements.txt
```

## Example for FAQ database

```python
import json

from database.builders import QdrantDatabaseBuilder
from database.qdrant import FAQQdrantDatabase, QdrantDatabase, SingletonQdrant
from embedder.tfidf import TfIdf


""" Get json with faq-objects like:
    [
        {
            "id": 1,
            "title": "this is a question",
            "description": "this is relevant answer"
        },
        ...
    ]

"""
with open("faqs.json", 'r', encoding="utf-8") as file:
    faq = json.load(file)

# Init vector database index
db_path = "vector_db"
index = SingletonQdrant(path=db_path)
database_builder = QdrantDatabaseBuilder(index=index)

# Create or load from file faq database
faq_name = "faq"
if os.path.exists(os.path.join(db_path, faq_name)):
    faq_database = FAQQdrantDatabase.load(
        name=faq_name,
        index=index,
        model=TfIdf(),
        questions_collection_name="questions",
        answers_collection_name="answers",
    )
else:
    faq_database = database_builder.build_faq_database(
        name=faq_name,
        faq_json=faq_first,
        id_field="id",
        question_field="title",
        answer_field="description",
        questions_collection_name="questions",
        answers_collection_name="answers",
        model=TfIdf(), 
    )

# Search for similar answers
# It returns list or Record objects with ids and content (Record.payload["content"])
responce = faq_database.search(query="text request example", limit=5)

# If faq changed you can update/delete content and vectors in DB
faq_database.update_vectors(faq_update)
faq_database.delete_vectors(ids=[1, 4, 5])

# And add new items of faq
faq_database.add_vectors(faq_new)

# To update embedding model (for better vectors) recalculate vector with accumulated content
faq_database.update_index()

# Also you can find duplicates of answers inside the database
# It returns list of set with ids of duplicated answers
duplicates = faq_database.find_duplicates(score_threshold=0.9)
"""
[{1, 34, 23}, {7, 24}]
"""

# Also you can try to find duplicate-like objects on request
query_duplicates = faq_database.search_similar(query="some text request")

```

## Example for sigle-source database
Single source DB has the same interfaces, only creating/loading process goes slightly different:

```python
collection_name = "test_documents"
if os.path.exists(os.path.join(db_path, collection_name)):
    test_database = QdrantDatabase.load(name=collection_name, index=index, model=TfIdf())
else:
    test_database = database_builder.build_database(
        name=collection_name,
        json_items=docs,
        id_field="id",
        content_field="text",
        collection_name=collection_name,
        model=TfIdf(),
    )

```
