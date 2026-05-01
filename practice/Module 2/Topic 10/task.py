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

def print_evaluation_results(query, response, result, evaluation_type):
    print(f"\n================ Automated Evaluation {evaluation_type} Results ================\n")
    print(f"Score: {result.score}")
    print("\n================================================\n")
    print(f"\n\n================ Manual Evaluation {evaluation_type} ================\n")
    print(f"Query: \n{query}")
    print(f"\n\nContext: \nCheck the knowledge_base.txt file for the context used in this evaluation.")
    print(f"\n\nResponse: \n{response}")
    print("\n================================================\n\n")

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

query="What projects has Sasidhar built?"

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="refine",
)
response = query_engine.query(query)

from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

results = faithfulness_evaluator.evaluate(
    response=response.response, 
    query=query,
    contexts=[n.text for n in nodes],
)


print_evaluation_results(query, response.response, results, evaluation_type="Faithfulness")

results = relevancy_evaluator.evaluate(
    response=response.response,
    query=query,
    contexts=[n.text for n in nodes],
)
print_evaluation_results(query, response.response, results, evaluation_type="Relevancy")
