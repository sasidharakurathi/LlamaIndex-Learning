import os

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

QUERY = "How does ATS scoring work?"


def build_documents():
	parent_doc = Document(
		text=(
			"ATS Overview\n\n"
			"The ATS is a Django-based application tracking system that connects the "
			"Resume Parser, Scoring Engine, and Face Recognition modules. "
			"It orchestrates resume ingestion, candidate ranking, and secure login."
		),
		metadata={
			"type": "project",
			"name": "ATS",
			"node_role": "parent",
		},
	)

	child_docs = [
		Document(
			text=(
				"Resume Parser\n\n"
				"Extracts text from PDF and DOCX resumes, cleans the content, and "
				"prepares structured candidate data for scoring."
			),
			metadata={
				"type": "module",
				"parent": "ATS",
				"name": "Resume Parser",
				"module": "resume_parser",
			},
		),
		Document(
			text=(
				"Scoring Engine\n\n"
				"Combines deterministic rules, keyword matching, and LLM-assisted "
				"signals to rank candidates for the ATS."
			),
			metadata={
				"type": "module",
				"parent": "ATS",
				"name": "Scoring Engine",
				"module": "scoring",
			},
		),
		Document(
			text=(
				"Face Recognition\n\n"
				"Handles biometric authentication using face encodings so admins can "
				"securely access the ATS dashboard."
			),
			metadata={
				"type": "module",
				"parent": "ATS",
				"name": "Face Recognition",
				"module": "face_recognition",
			},
		),
	]

	return [parent_doc, *child_docs]


def print_nodes(title, nodes):
	print("=" * 72)
	print(title)
	print("=" * 72)

	for index, node_with_score in enumerate(nodes, start=1):
		node = node_with_score.node
		score = getattr(node_with_score, "score", None)

		print(f"--- Node {index} ---")
		if score is not None:
			print(f"Score: {score}")
		print(f"Node ID: {node.node_id}")
		print(f"Metadata: {node.metadata}")
		print(f"Text: {node.get_content()}")
		print()


def build_answer(llm, nodes, query):
	context = "\n\n".join(node.node.get_content() for node in nodes)
	prompt = f"""
You are answering from the retrieved ATS context only.

Context:
{context}

Question:
{query}

Answer in 2-3 concise sentences.
"""

	response = llm.complete(prompt)
	return response.text if hasattr(response, "text") else str(response)


def main():
	documents = build_documents()

	embed_model = HuggingFaceEmbedding(
		model_name="BAAI/bge-small-en",
		device="cpu",
	)

	llm = GoogleGenAI(
		model="gemini-2.5-flash",
		api_key=GEMINI_API_KEY,
	)

	index = VectorStoreIndex.from_documents(
		documents=documents,
		embed_model=embed_model,
	)

	print("QUERY:")
	print(QUERY)

	initial_retriever = index.as_retriever(similarity_top_k=1)
	initial_nodes = initial_retriever.retrieve(QUERY)
	print_nodes("INITIAL RETRIEVAL", initial_nodes)

	parent_node = initial_nodes[0].node if initial_nodes else None
	if parent_node is None:
		print("No parent node was retrieved, so recursive expansion stopped.")
		return

	child_filters = MetadataFilters(
		filters=[ExactMatchFilter(key="parent", value="ATS")]
	)
	child_retriever = index.as_retriever(
		filters=child_filters,
		similarity_top_k=3,
	)
	expanded_nodes = child_retriever.retrieve(QUERY)
	print_nodes("RECURSIVE EXPANSION", expanded_nodes)

	answer = build_answer(llm, [initial_nodes[0], *expanded_nodes], QUERY)
	print("=" * 72)
	print("FINAL ANSWER")
	print("=" * 72)
	print(answer)


if __name__ == "__main__":
	main()
