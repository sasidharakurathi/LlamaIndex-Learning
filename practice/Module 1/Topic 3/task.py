# --- Try to do this task by yourself first, and then check the solution. ---

from dotenv import load_dotenv
import os
from pprint import pprint
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from datetime import datetime

start = datetime.now()

from llama_index.llms.google_genai  import GoogleGenAI
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
    device="cpu"
)

from llama_index.core import Document

documents = [
    Document(text="# Sasidhar Akurathi - AI Clone Knowledge Base\n\n## Profile\n\n**Full Name**: Sasidhar Akurathi\n\n**Headline**: Building Scalable FastAPI and Django Backends | Orchestrating LLM Agents with n8n and Google ADK | AI Automation & DevOps\n\n**Current Role**: Final-year Computer Science Student | Backend & AI Engineering Intern\n\n### Education\n- **Degree**: B.Tech Computer Science\n- **College**: P.S.C.M.R College of Engineering & Technology\n- **Year**: 2022 - 2026\n- **Cgpa**: 8.5\n\n**Permanent Location**: Vijayawada, Andhra Pradesh, India\n\n**Current Location**: Hyderabad, Telangana, India"),
    Document(text="### Emails\n- akurathisasidhar4@gmail.com\n\n**Social Links**: {'linkedin': 'https://www.linkedin.com/in/sasidhar-akurathi', 'github': 'https://github.com/sasidharakurathi', 'portfolio': 'https://sasidharakurathi-portfolio.vercel.app/'}\n\n---\n\n## Summary\n\n**Short**: AI-focused software developer with expertise in backend systems, computer vision, and building intelligent desktop and web applications.\n\n**Detailed**: A dynamic software engineer with a strong focus on Artificial Intelligence and full-stack development. Demonstrated experience in creating AI-driven solutions, such as the VirtuHire AI recruitment system and automated interviewers, as well as robust desktop utilities like RoboSwift and Vega using modern frameworks like Tauri. Proficient across multiple programming languages including Python, TypeScript, Rust, and HTML/JS, with a solid foundation in deep learning, data analysis, and workflow automation.\n\n### Interests\n- Artificial Intelligence\n- Backend Development\n- Cybersecurity\n- DevOps and System Design\n- Building Scalable Web Applications\n- Real-time Systems\n- Computer Vision\n- LLM and Generative AI\n- Automation and Productivity Tools\n\n**Career Goal**: To leverage expertise in AI and backend technologies to build innovative, scalable, and intelligent software solutions that solve real-world problems."),
]

from llama_index.core.node_parser import SentenceSplitter

node_parser = SentenceSplitter(
    chunk_size=50,
    chunk_overlap=20
)

nodes = node_parser.get_nodes_from_documents(documents)

# --- 1. VectorStoreIndex ---
# from llama_index.core import VectorStoreIndex

# index  = VectorStoreIndex(
#     nodes=nodes,
#     embed_model=embed_model
# )

# --- 2. SummaryIndex ---
# from llama_index.core import SummaryIndex

# index = SummaryIndex(
#     nodes=nodes,
#     embed_model=embed_model
# )

# --- 3. TreeIndex ---
from llama_index.core import TreeIndex

index = TreeIndex(
    nodes=nodes,
    embed_model=embed_model,
    llm=llm
)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What is Sasidhar Email?")
pprint(response)

retriever = index.as_retriever()

nodes = retriever.retrieve("What is Sasidhar Email?")

for i, node in enumerate(nodes):
    print(f"\n--- Node {i+1} ---")
    print("Score:", node.score)
    print("Text:", node.text)
    
stop = datetime.now()

print("\nTime taken:", (stop - start).total_seconds(), "seconds")