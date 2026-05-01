import re
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def test_response_sysntesis_modes(query, index, llm, response_mode):
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode=response_mode,
    )
    response = query_engine.query(query)
    
    print(f"\n\n================ Response of {response_mode} Mode ================\n\n")
    print(response)
    print("\n\n================================================\n\n")

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
)

knowledge_base = ""
with open("./knowledge_base.txt", "r", encoding="utf-8") as file:
    knowledge_base = clean_text(file.read())

documents = [
    Document(text=knowledge_base),
]

sentence_splitter_parser = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

nodes = sentence_splitter_parser.get_nodes_from_documents(documents=documents)

index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
)

# Try test queries as well and oberve the differences in the responses based on the response synthesis mode used.
# query="What are Sasidhar's key strengths and projects?"
# query="Give a detailed evaluation of Sasidhar's ATS system"
query="What projects has Sasidhar built?"

test_response_sysntesis_modes(
    query=query,
    index=index,
    llm=llm,
    response_mode="compact"
)

test_response_sysntesis_modes(
    query=query,
    index=index,
    llm=llm,
    response_mode="refine"
)

test_response_sysntesis_modes(
    query=query,
    index=index,
    llm=llm,
    response_mode="tree_summarize"
)