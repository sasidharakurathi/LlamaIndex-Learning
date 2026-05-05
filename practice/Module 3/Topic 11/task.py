from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
).load_data()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SemanticSplitterNodeParser

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
)

semantic_splitter_parser = SemanticSplitterNodeParser(
    embed_model=embed_model, buffer_size=1, breakpoint_percentile_threshold=95
)

nodes = semantic_splitter_parser.get_nodes_from_documents(documents=documents)

index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
    
)

retirever = index.as_retriever()

query = "What projects has Sasidhar built?"

nodes = retirever.retrieve(query)

query_engine = index.as_query_engine(
    llm=llm,
)

response = query_engine.query(
    query,
)

print(f"\nQuery: {query}\n")
print(f"\nNodes used for response: {[node.get_content() for node in nodes]}\n")
print(f"\nResponse: {response.response}\n")
