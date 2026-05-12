from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI


llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
)

documents = SimpleDirectoryReader(
    input_files=[
        "./profile_kb.txt",
        "./projects_kb.txt",
    ]
).load_data()

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)

portfolio_query_engine = index.as_query_engine(llm=llm)

portfolio_tool = QueryEngineTool.from_defaults(
    query_engine=portfolio_query_engine,
    name="portfolio_query_tool",
    description=(
        "Searches Sasidhar's profile and projects to answer questions about "
        "what he has built, his background, and his experience."
    ),
)

agent = ReActAgent(
    tools=[portfolio_tool],
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are an assistant that STRICTLY answers questions about Sasidhar's projects and experience.\n"
        "CRITICAL RULES:\n"
        "1. You MUST call portfolio_query_tool exactly once before giving a final answer.\n"
        "2. Do not answer from memory or prior knowledge.\n"
        "3. Final answer must be based only on the tool output.\n"
        "4. If the tool output lacks the requested information, say it is not in the knowledge base."
    ),
)

async def main():
    print("===================================================================")
    print('Querying the ReAct agent: "What projects has Sasidhar built?"')
    print("===================================================================")

    prompts = [
        "What projects has Sasidhar built? Use portfolio_query_tool before answering.",
        "Mandatory: Call portfolio_query_tool first, then answer using only its output. What projects has Sasidhar built?",
        "You must call portfolio_query_tool now. Do not answer directly. Query: What projects has Sasidhar built?",
    ]

    response = None
    for prompt in prompts:
        handler = agent.run(user_msg=prompt)
        tool_called = False

        async for ev in handler.stream_events():
            if ev.__class__.__name__ == "ToolCall":
                tool_called = True

        result = await handler
        if tool_called:
            response = result
            break

    if response is None:
        raise RuntimeError("Agent did not call portfolio_query_tool in any attempt.")

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
