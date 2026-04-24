import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

documents = [
    Document(text="# Sasidhar Akurathi - AI Clone Knowledge Base\n\n## Profile\n\n**Full Name**: Sasidhar Akurathi\n\n**Headline**: Building Scalable FastAPI and Django Backends | Orchestrating LLM Agents with n8n and Google ADK | AI Automation & DevOps\n\n**Current Role**: Final-year Computer Science Student | Backend & AI Engineering Intern\n\n### Education\n- **Degree**: B.Tech Computer Science\n- **College**: P.S.C.M.R College of Engineering & Technology\n- **Year**: 2022 - 2026\n- **Cgpa**: 8.5\n\n**Permanent Location**: Vijayawada, Andhra Pradesh, India\n\n**Current Location**: Hyderabad, Telangana, India"),
    Document(text="### Emails\n- akurathisasidhar4@gmail.com\n\n**Social Links**: {'linkedin': 'https://www.linkedin.com/in/sasidhar-akurathi', 'github': 'https://github.com/sasidharakurathi', 'portfolio': 'https://sasidharakurathi-portfolio.vercel.app/'}\n\n---\n\n## Summary\n\n**Short**: AI-focused software developer with expertise in backend systems, computer vision, and building intelligent desktop and web applications.\n\n**Detailed**: A dynamic software engineer with a strong focus on Artificial Intelligence and full-stack development. Demonstrated experience in creating AI-driven solutions, such as the VirtuHire AI recruitment system and automated interviewers, as well as robust desktop utilities like RoboSwift and Vega using modern frameworks like Tauri. Proficient across multiple programming languages including Python, TypeScript, Rust, and HTML/JS, with a solid foundation in deep learning, data analysis, and workflow automation.\n\n### Interests\n- Artificial Intelligence\n- Backend Development\n- Cybersecurity\n- DevOps and System Design\n- Building Scalable Web Applications\n- Real-time Systems\n- Computer Vision\n- LLM and Generative AI\n- Automation and Productivity Tools\n\n**Career Goal**: To leverage expertise in AI and backend technologies to build innovative, scalable, and intelligent software solutions that solve real-world problems."),
]

node_parser = SentenceSplitter(
    chunk_size=50,
    chunk_overlap=20
)

nodes = node_parser.get_nodes_from_documents(documents)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
    device="cpu"
)

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    # First time: build index
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load existing index
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR
    )
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What has Sasidhar built?")
print(response)