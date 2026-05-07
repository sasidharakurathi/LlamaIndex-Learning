from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from llama_index.core import VectorStoreIndex, SummaryIndex, Document
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Helper function: To load documents from a text file.
def load_documents(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return [
        Document(
            text=text,
        )
    ]
    
# Helper function: To print all nodes in the index.
def print_all_nodes(nodes):
    print("===================================================================")
    for node in nodes:
        print(f"--- Node ID: {node.node_id}: ---\n")
        print(node.get_content())
    print("\n\n===================================================================")

def query_knowledge_base(router, query, filters=None):
    print(f"Query: {query}")
    response = router.query(query)
    
    if hasattr(response, 'metadata') and response.metadata:
        print(f"Selected Tool Info: {response.metadata}")
    
    print("Response to query:\n")
    print(response.response)
    print("\n\n===================================================================")
   

llm = GoogleGenAI(
	model="gemini-2.5-flash",
	api_key=GEMINI_API_KEY,
)

embebd_model = HuggingFaceEmbedding(
	model_name="BAAI/bge-small-en",
)

profile_doc = load_documents("./profile_kb.txt")
project_doc = load_documents("./projects_kb.txt")

vector_index = VectorStoreIndex.from_documents(
    project_doc,
    embed_model=embebd_model,
)

summary_index = SummaryIndex.from_documents(
    profile_doc,
	llm=llm,
)

vector_engine = vector_index.as_query_engine(
	llm=llm,
	response_mode="compact",
)

summary_engine = summary_index.as_query_engine(
	llm=llm,
	response_mode="tree_summarize",
)

vector_tool = QueryEngineTool.from_defaults(
	query_engine=vector_engine,
	description="Useful for factual questions about projects, technologies, and implementation details.",
)

summary_tool = QueryEngineTool.from_defaults(
	query_engine=summary_engine,
	description="Useful for summarizing experience, profiles, and overall overviews.",
)

router = RouterQueryEngine(
	selector=LLMSingleSelector.from_defaults(
		llm=llm,
	),
	llm=llm,
 query_engine_tools=[
	 vector_tool,
	 summary_tool,
 ]
)

print("===================================================================")
print("Querying the router `Give an overview of Sasidhar's experience`...")
print("===================================================================")
query_knowledge_base(router, "Give an overview of Sasidhar's experience")

print("===================================================================")
print("Querying the router `What technologies were used in ATS?`...")
print("===================================================================")
query_knowledge_base(router, "What technologies were used in ATS?")

