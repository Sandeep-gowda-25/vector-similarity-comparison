import os
import pandas as pd
import redis
import numpy as np

from redis.commands.search.field import VectorField, TextField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import ast
from typing import Literal,List
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI,status,HTTPException

app = FastAPI()

def get_redis_client():
    r = redis.Redis(host='redis-app', port=6379, db=0,password="mypassword")
    return r

def create_index(conn):
    VECTOR_DIMENSION = 1536
    VECTOR_DIMENSION_LOWER = 384
    index_name = "test"
    SCHEMA = (
        TextField("text"),
        VectorField(
            "content_embeddings_KNN",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSION,
                "DISTANCE_METRIC": "COSINE",
            }
        ),
        VectorField(
            "content_embeddings_ANN",
            "HNSW",
            {
                "TYPE":"FLOAT32",
                "DIM":VECTOR_DIMENSION,
                "DISTANCE_METRIC":"COSINE"
            }
        ),
        VectorField(
            "content_embeddings_KNN_hf",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSION_LOWER,
                "DISTANCE_METRIC": "COSINE",
            }
        ),
        VectorField(
            "content_embeddings_ANN_hf",
            "HNSW",
            {
                "TYPE":"FLOAT32",
                "DIM":VECTOR_DIMENSION_LOWER,
                "DISTANCE_METRIC":"COSINE"
            }
        )
    )
    definition = IndexDefinition(prefix=["text:"], index_type=IndexType.HASH)
    PREFIX = "text:"
    INDEX_NAME = index_name
    # Check if index exists
    try:
        conn.ft("test").info()
        print("Index already exists")
    except:
        # Create RediSearch Index
        conn.ft("test").create_index(
            fields = SCHEMA,
            definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
        )

    info = conn.ft(INDEX_NAME).info()
    print(f'{info=}')
    num_docs = info["num_docs"]
    indexing_failures = info["hash_indexing_failures"]
    print(f"{num_docs} documents indexed with {indexing_failures} failures")

    return f"{num_docs} documents indexed with {indexing_failures} failures"

@app.get('/index')
def index_exists():
    # Get Redis Client
    conn=get_redis_client()
    try:
        conn.ft("test").info()
        return True
    except:
        return False

def add_data(client,chunks,embeddings,embeddings_lower_dim):
    pipeline = client.pipeline(transaction=False)
    for index, (chunk, embedding,embedding_hf) in enumerate(zip(chunks, embeddings,embeddings_lower_dim)):
        key = (
            "text:"
            + str(index).strip()
        )
        print(chunk)

        pipeline.hset(
            key,
            mapping={
                "text": chunk,
                "content_embeddings_KNN": np.array(
                    # ast.literal_eval(embedding), dtype=np.float32
                    embedding, dtype=np.float32
                ).tobytes(),
                "content_embeddings_ANN": np.array(
                    # ast.literal_eval(embedding), dtype=np.float32
                    embedding, dtype=np.float32
                ).tobytes(),
                "content_embeddings_KNN_hf": np.array(
                    embedding_hf, dtype=np.float32
                ).tobytes(),
                "content_embeddings_ANN_hf": np.array(
                    embedding_hf, dtype=np.float32
                ).tobytes(),
            },
        )
    pipeline.execute()
    print(
        f"{index=} Completed writing to Redis for file"
    )

@app.get('/add_data')
def setup():
    try:
        data = pd.read_csv('redis_test.csv')
        texts = data["text"]
        embeddings = get_embeddings(texts=texts)#to get high dimensional embeddings from openai ada model
        embeddings_lower_dim = get_text_embeddings_384(texts=texts)
        conn = get_redis_client()
        index_there = index_exists()
        if index_there==True:
            cleanup(conn)
        create_index(conn=conn)
        add_data(conn,texts,embeddings,embeddings_lower_dim)
        return "Successfuly added data to redis"
    except Exception as e:
        return (f"Error - {e}"),status.HTTP_500_INTERNAL_SERVER_ERROR

def cleanup(conn)->None:
    return conn.execute_command('FT.DROPINDEX', 'test', 'DD')

from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv(".env")
import os

base_url = os.getenv("OPENAI_ENDPOINT")
token = os.getenv("OPENAI_API_KEY")
client = AzureOpenAI(azure_endpoint=base_url,
                    api_key=token,
                     api_version="2023-09-15-preview"
                    )
def get_embeddings(texts:List[str]):
    embeddings = []
    for text in texts:
        embeddings.append(client.embeddings.create(model=os.getenv("EMBEDDING_MODEL"),
                        input=text).data[0].embedding)
    return embeddings


@app.get('/search')
def search_vector(text:str,algorithm:Literal['KNN','ANN'],dimension:Literal["higher","lower"],k=5):
    try:
        print(text)
        if dimension=="higher":
            input_vector = np.array(get_embeddings([text])[0],dtype=np.float32).tobytes()
        else:
            input_vector = np.array(get_text_embeddings_384([text])[0],dtype=np.float32).tobytes()
        if algorithm=='KNN' and dimension=='higher':
            base_query = f'(*)=>[KNN {k} @content_embeddings_KNN $query_vector AS vector_score]'
        elif algorithm=='ANN' and dimension=='higher':
            base_query = f'(*)=>[KNN {k} @content_embeddings_ANN $query_vector AS vector_score]'
        elif algorithm=='KNN' and dimension=='lower':
            base_query = f'(*)=>[KNN {k} @content_embeddings_KNN_hf $query_vector AS vector_score]'
        elif algorithm=='ANN' and dimension=='lower':
            base_query = f'(*)=>[KNN {k} @content_embeddings_ANN_hf $query_vector AS vector_score]'
        params_dict = {"query_vector": input_vector}
        conn = get_redis_client()
        results = conn.ft("test").search(get_vector_query_object(base_query,k), params_dict)
        # results = conn.ft("test").search(query=base_query,query_params=params_dict)
        return results
    except Exception as e:
        return (f"Error occured : {e}"),status.HTTP_500_INTERNAL_SERVER_ERROR

def get_vector_query_object(base_query, k=3):
    query = (
    Query(base_query)
        .return_fields(*["text", "vector_score"])
        .paging(0,k)
        .sort_by("vector_score")
        .dialect(2))
    # print(query.query_string())
    return query

def get_text_embeddings_384(texts:List[str]):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings