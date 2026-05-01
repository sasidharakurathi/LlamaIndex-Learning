from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

queries = ["What has Sasidhar built?", "List his projects", "What are his works?"]

for query in queries:
    print(f"Query: {query}")
    print(f"Embedding: {embed_model.get_query_embedding(query)}\n")

