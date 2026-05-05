from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

KNOWLEDGE_BASE = "./knowledge_base.txt"

def clean_knowledge_base(knowledge_base: str) -> str:
    raw_document = []
    result = ""
    with open(knowledge_base, "r", encoding="utf-8") as f:
        raw_documents = f.readlines()
    
    print(f"Loaded {len(raw_documents)} Raw documents")
    print("Cleaning documents...")
    
    for document in raw_documents:
        cleaned_document = document.replace("#", "").replace("*", "").strip()
        result += cleaned_document + "\n"
    
    with open("./cleaned_knowledge_base.txt", "w", encoding="utf-8") as f:
        f.write(result)
    
    return result
    
cleaned_documents = clean_knowledge_base(KNOWLEDGE_BASE)

from llama_index.core import Document, VectorStoreIndex
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

semantic_splitter_parser = SemanticSplitterNodeParser(
    llm=llm,
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model,
)

documents = [
    Document(
        text=cleaned_documents,
        metadata={
            "type": "project",
            "name": "ATS",
            "domain": "AI"
        }
    ),
]

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


