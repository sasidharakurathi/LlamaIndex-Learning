from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)

from llama_index.core import VectorStoreIndex, Document

import re

def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def print_context(retriever, query, retriever_name):
    context = retriever.retrieve(query)
    print(f"\n\n================ CONTEXT OF {retriever_name} ================\n\n")
    for i, c in enumerate(context):
        print(f"\n--- Node {i+1} ---")
        print("Score:", c.score)
        print("Text:", c.text)
    print("\n\n================================================\n\n")


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

knowledge_base = ""
with open("./knowledge_base.txt", "r", encoding="utf-8") as f:
    knowledge_base = clean_text(f.read())

documents = [Document(text=knowledge_base)]

llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)

semantic_splitter_parser = SemanticSplitterNodeParser(
    embed_model=embed_model, buffer_size=1, breakpoint_percentile_threshold=95
)
nodes = semantic_splitter_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
similarity_retriever = index.as_retriever(similarity_top_k=3)
mmr_retirever = index.as_retriever(
    similarity_top_k=5,
    vector_store_query_mode="mmr",
    mmr_threshold=0.5
)

# Top-k similarity retriever
print_context(
    similarity_retriever,
    "ATS system features",
    "Similarity Retriever",
)

# MMR retriever
print_context(
    mmr_retirever,
    "ATS system features",
    "MMR Retriever",
)
