from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
)

from llama_index.core import VectorStoreIndex, Document

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

knowledge_base = ""
with open("knowledge_base.txt", "r") as f:
    knowledge_base = f.read()

documents = [Document(text=knowledge_base)]

llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)


# --- Chunking Stratergy-1: Sentence Splitter ---
sentence_splitter_parser = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=20,
)
sentence_splitter_nodes = sentence_splitter_parser.get_nodes_from_documents(documents)

# --- Chunking Stratergy-2: Semantic Splitter ---
semantic_splitter_parser = SemanticSplitterNodeParser(
    embed_model=embed_model, buffer_size=1, breakpoint_percentile_threshold=95
)
semantic_splitter_nodes = semantic_splitter_parser.get_nodes_from_documents(documents)


# --- Chunking Stratergy-3: Sentence Splitter (Sliding Window) ---
sentence_window_parser = SentenceWindowNodeParser(window_size=3)
sentence_window_nodes = sentence_window_parser.get_nodes_from_documents(documents)

nodes = [sentence_splitter_nodes, semantic_splitter_nodes, sentence_window_nodes]


def retrieve_and_print_context(nodes, query, chunker_name):
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
    retriever = index.as_retriever()
    context = retriever.retrieve(query)
    print(f"\n\n================ CONTEXT OF {chunker_name} ================\n\n")
    for i, c in enumerate(context):
        print(f"\n--- Node {i+1} ---")
        print("Score:", c.score)
        print("Text:", c.text)
    print("\n\n================================================\n\n")


# --- Benchmarking two Chunking Stratergies ---
retrieve_and_print_context(
    sentence_splitter_nodes, "What projects has Sasidhar built?", "Sentence Splitter"
)
retrieve_and_print_context(
    semantic_splitter_nodes, "What projects has Sasidhar built?", "Semantic Splitter"
)
retrieve_and_print_context(
    sentence_window_nodes,
    "What projects has Sasidhar built?",
    "Sentence Window Splitter",
)
