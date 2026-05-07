from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
)

documents = SimpleDirectoryReader(
    input_files=[
        "./profile_kb.txt",
        "./projects_kb.txt"
    ]
).load_data()


parser = SemanticSplitterNodeParser(
    embed_model=embed_model,
    breakpoint_percentile_threshold=95
)

nodes = parser.get_nodes_from_documents(documents)


graph_index = PropertyGraphIndex(
    nodes=nodes,
    llm=llm,
    embed_model=embed_model,
    max_triplets_per_chunk=10,
) 
    
query_engine = graph_index.as_query_engine(llm=llm)

response = query_engine.query(
    "What technologies are used in ATS?"
)

print("===================================================================")
print(response.response)
print("===================================================================")