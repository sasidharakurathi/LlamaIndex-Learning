from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

PROJECTS_KB = "./projects_kb.txt"
EXPERIENCE_KB = "./experience_kb.txt"

# Helper function: To load documents from a text file.
def load_documents(file_path, metadata=None):
    with open(file_path, "r") as f:
        text = f.read()
    return [
        Document(
            text=text,
            metadata=metadata
        )
    ]

# Helper function: To print all nodes in the index.
def print_all_nodes(nodes):
    print("===================================================================")
    for node in nodes:
        print(f"--- Node ID: {node.node_id}: ---\n")
        print(node.get_content())
    print("\n\n===================================================================")

# Helper function: To query the knowledge base and print the response.
def query_knowledge_base(index, query, filters=None):
    query_engine = index.as_query_engine(llm=llm, filters=filters)
    response = query_engine.query(query)
    print("Response to query:\n")
    print(response.response)
    print("\n\n===================================================================")
    
    
def retrieve_nodes(index, filters):
    retriever = index.as_retriever(filters=filters)
    retrieved_nodes = retriever.retrieve("What projects has Sasidhar built?")
    print_all_nodes(retrieved_nodes)

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
)

semantic_splitter_parser = SemanticSplitterNodeParser(
    embed_model=embed_model, buffer_size=1, breakpoint_percentile_threshold=95
)

project_docs = load_documents(PROJECTS_KB, {"type": "project"})
experience_docs = load_documents(EXPERIENCE_KB, {"type": "experience"})

project_filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="type", value="project")
    ]
)

experience_filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="type", value="experience")
    ]
)

index = VectorStoreIndex.from_documents(
    documents=project_docs + experience_docs,
    embed_model=embed_model,
)

nodes = index.storage_context.docstore.docs.values()

print("===================================================================")
print("All Knowledge Base Nodes:")
print_all_nodes(nodes)

print("Querying the knowledge base without any filters...")
print("===================================================================")
query_knowledge_base(index, "What projects has Sasidhar built?")

print("Retrieving Nodes with project filters...")
retrieve_nodes(index, project_filters)

print("Querying the knowledge base with project filters...")
print("===================================================================")
query_knowledge_base(index, "What projects has Sasidhar built?", filters=project_filters)


print("Retrieving Nodes with experience filters...")
retrieve_nodes(index, experience_filters)

print("Querying the knowledge base with experience filters...")
print("===================================================================")

query_knowledge_base(index, "What is Sasidhar's experience?", filters=experience_filters)

