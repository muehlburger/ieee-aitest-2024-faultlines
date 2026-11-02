"""Embed the normal-operation rows and store them in Qdrant.

Start Qdrant first:
    docker run -p 6333:6333 qdrant/qdrant
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from itertools import islice
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEASUREMENTS = DATA_DIR / "battery_healthy_roundend_and_text_encoded_with_is_num2words.csv"
COLLECTION_NAME = "measurements"
NUM_OF_ROWS_TO_READ = 11000

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--host", default="localhost", help="Qdrant host")
parser.add_argument("--port", default=6333, type=int, help="Qdrant port")
args = parser.parse_args()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(host=args.host, port=args.port, timeout=60 * 60)


def make_collection(client, collection_name):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )


def make_chunks(input_text):
    df = pd.read_csv(input_text, header=None, names=["measurements"], sep=",", nrows=NUM_OF_ROWS_TO_READ)
    print(f"Number of rows: {len(df)}")
    data = DataFrameLoader(df, page_content_column="measurements").load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        length_function=len,
        add_start_index=False,
        separators="\n\n",
    )
    return text_splitter.split_documents(data)


def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def upsert_to_qdrant(vectors, texts):
    payloads = [{"text": item.page_content} for item in texts]
    vectors = np.array(vectors).tolist()
    for batch in tqdm(batched(zip(vectors, payloads), 100)):
        batch_vectors, batch_payloads = zip(*batch)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=[str(uuid.uuid4()) for _ in range(len(batch_vectors))],
                vectors=batch_vectors,
                payloads=batch_payloads,
            ),
        )


make_collection(client, COLLECTION_NAME)
texts = make_chunks(MEASUREMENTS)
vectors = embeddings.embed_documents([item.page_content for item in texts])
upsert_to_qdrant(vectors, texts)
