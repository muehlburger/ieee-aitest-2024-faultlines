"""Ask a local LLM whether each test row contains an anomaly.

The ten most similar normal rows are retrieved from Qdrant and given to the
model as context. Answers are written to results/.
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
from pathlib import Path
import warnings

import pandas as pd
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from tqdm import tqdm

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENTS = ROOT / "data" / "battery_runtime_fault1_rounded_and_text_encoded_with_is_num2words.csv"
RESULTS_DIR = ROOT / "results"
COLLECTION_NAME = "measurements"
NUM_OF_DOCUMENTS = 10

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--start", default=3900, type=int, help="Start index of the time series")
parser.add_argument("-e", "--end", default=4100, type=int, help="Stop index of the time series")
parser.add_argument("-m", "--model", default="mixtral", help="Ollama model (mistral, mixtral, gemma:instruct, command-r, codellama, qwen:14b, mistral-openorca, dolphin-mixtral)")
parser.add_argument("--host", default="localhost", help="Qdrant host")
parser.add_argument("--port", default=6333, type=int, help="Qdrant port")
parser.add_argument("--device", default="cpu", help="Device for the embedding model and Ollama, e.g. cuda:0")
args = parser.parse_args()

logging.basicConfig(filename="std.log", format="%(asctime)s %(message)s", filemode="w")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": args.device})
client = QdrantClient(host=args.host, port=args.port, timeout=60)
qdrant = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
    metadata_payload_key="metadata",
    content_payload_key="text",
)

df = pd.read_csv(MEASUREMENTS, header=None, names=["measurements"])
data = DataFrameLoader(df, page_content_column="measurements").load()

llm = ChatOllama(model=args.model, temperature=0, device=args.device)

template = """
Use the following context to answer the question at the end. If you cannot determine, just say no, don't try to make up an answer. Use short, technical answers like (zero anomalies, yes, no). Write a response that appropriately completes the request.

Context: Normal operating range:
{context}

Question: {question}

Test Measurement:
{measurement}

Provide an anomaly rating between 0-100 based on the following criteria:

1) Values are in a reasonable range of the provided context.
2) Values do not contain anomalies.
3) Values are consistent with the given context.

Check *very carefully* with the given ranges before giving your rating.

Answer:
- Anomaly: e.g. yes, no
- Anomaly-Rating:
- Explanation:

"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": NUM_OF_DOCUMENTS})
chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT, verbose=False)

results = {}
reason = {}
lines = range(args.start, args.end)
question = "Do the following measurements contain anomalies? Yes or no?"

for i in tqdm(lines):
    measurement = data[i].page_content.strip()
    docs = retriever.get_relevant_documents(measurement, return_source_documents=False)
    result = chain.invoke(
        {"input_documents": docs, "question": question, "measurement": measurement},
        return_only_outputs=True,
    )
    answer = result["output_text"].strip()
    logger.info(f"Line {i}: {answer}")
    reason[i] = answer
    results[i] = "yes" in answer.lower()

out = pd.DataFrame(data={"line": lines, "anomaly": [results[i] for i in lines], "reason": [reason[i] for i in lines]})
out.to_csv(RESULTS_DIR / f"results_from_{args.start}_to_{args.end}_{args.model}.csv", index=False)
