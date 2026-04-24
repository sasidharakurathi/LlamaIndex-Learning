
# Step 0: Load and setup GEMINI_API_KEY
from dotenv import load_dotenv
import os
from pprint import pprint
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# Step 1: Configure LLM (Gemini)
from llama_index.llms.google_genai  import GoogleGenAI
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

# Step 2: Embedding Model (Local)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
    device="cpu"
)

# Step 3: Create Documents
from llama_index.core import Document

documents = [
    Document(text="Sasidhar is an AI engineer specializing in backend systems."),
    Document(text="He built an ATS system with resume parsing and LLM."),
]

# Step 4: Create Index
from llama_index.core import VectorStoreIndex

index  = VectorStoreIndex.from_documents(
    documents=documents,
    embed_model=embed_model,
)

# Step 5: Query Engine
query_engine = index.as_query_engine(llm=llm)

# Step 6: Ask
response = query_engine.query("What has Sasidhar built?")
pprint(response)

retriever = index.as_retriever()

# Step 7: Debugging
nodes = retriever.retrieve("What has Sasidhar built?")

for i, node in enumerate(nodes):
    print(f"\n--- Node {i+1} ---")
    print("Score:", node.score)
    print("Text:", node.text)