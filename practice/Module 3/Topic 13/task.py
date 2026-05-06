from dotenv import load_dotenv
import os
import shutil

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PERSIS_DIR = "./storage"
PRIMARY_KNOWLEDGE_BASE = "./primary_knowledge_base.txt"
NEW_KNOWLEDGE_BASE = "./new_knowledge_base.txt"

# Remove existing persistence directory if it exists to start fresh
if os.path.exists(PERSIS_DIR):
    print("Removing Exsiting Persistence Directory...\n")
    shutil.rmtree(PERSIS_DIR)

# Helper function: To load documents from a text file.
def load_documents(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return [Document(text=text)]

# Helper function: To print all nodes in the index.
def print_all_nodes(index):
    print("===================================================================\n\n")
    for node in index.storage_context.docstore.docs.values():
        print(f"--- Node ID: {node.node_id}: ---\n")
        print(node.get_content())
    print("\n\n===================================================================\n\n")

# Helper function: To insert new nodes into the index and persist the changes.
def insert_new_kb_nodes(index, new_nodes):    
    index.insert_nodes(new_nodes)
    index.storage_context.persist(persist_dir=PERSIS_DIR)
    return index

# Helper function: To delete nodes from the index and persist the changes.
def delete_kb_nodes(index, nodes_ids):
    index.delete_nodes(nodes_ids, delete_from_docstore=True)
    index.storage_context.persist(persist_dir=PERSIS_DIR)
    return index

# Helper function: To query the knowledge base and print the response.
def query_knowledge_base(query_engine, query):
    response = query_engine.query(query)
    print("Response to query:\n")
    print(response.response)


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

if not os.path.exists(PERSIS_DIR):
    primary_docs = load_documents(PRIMARY_KNOWLEDGE_BASE)
    primary_nodes = semantic_splitter_parser.get_nodes_from_documents(primary_docs)
    index = VectorStoreIndex(primary_nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=PERSIS_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIS_DIR)
    index = load_index_from_storage(storage_context)

print("Primary Knowledge Base Nodes:")
print_all_nodes(index)

print("Querying the primary knowledge base...\n")
query_engine = index.as_query_engine(llm=llm)
query_knowledge_base(query_engine, "What projects has Sasidhar built?")

print("===================================================================\n\n")

primary_kb_node_ids = [node.node_id for node in index.storage_context.docstore.docs.values()]


# Here we are inserting new nodes from the new knowledge base.
new_docs = load_documents(NEW_KNOWLEDGE_BASE)
new_nodes = semantic_splitter_parser.get_nodes_from_documents(new_docs)
index = insert_new_kb_nodes(index, new_nodes)

# Check the nodes after insertion to verify the new nodes are added.
print("Updated Knowledge Base Nodes:")
print_all_nodes(index)

# And we are querying the knowledge base again to see how the response has changed 
# after adding new nodes from the new knowledge base.
print("Querying the updated knowledge base (after adding new_knowledge_base.txt) ...\n")
query_engine = index.as_query_engine(llm=llm)
query_knowledge_base(query_engine, "What projects has Sasidhar built?")

print("===================================================================\n\n")

# Here we are deleting the primary knowledge base nodes to see how the response changes 
# when we remove the original knowledge base and only have the new knowledge base in the index.
print("Deleing primary knowledge base nodes...\n")
index = delete_kb_nodes(index, primary_kb_node_ids)

# Check the nodes after deletion to verify the primary knowledge base nodes are removed.
print("Knowledge Base Nodes after deletion:")
print_all_nodes(index)

# And we are querying the knowledge base again to see how the response has changed after 
# deleting the primary knowledge base nodes.
print("Querying the updated knowledge base (after deleting primary_knowledge_base.txt) ...\n")
query_engine = index.as_query_engine(llm=llm)
query_knowledge_base(query_engine, "What projects has Sasidhar built?")
