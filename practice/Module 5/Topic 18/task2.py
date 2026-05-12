from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI


llm = GoogleGenAI(
	model="gemini-2.5-flash",
	api_key=GEMINI_API_KEY,
)


def get_skills():
	full_list = [
		"Python",
		"Django",
		"FastAPI",
		"Google Gemini",
		"Ollama",
		"LlamaIndex",
		"Face Recognition",
		"PyPDF2",
		"python-docx",
		"MySQL",
		"WebSockets",
		"Tailwind CSS",
		"Qdrant",
		"BM25",
		"Hybrid Search",
	]
	# Make the list absolutely explicit
	return f"COMPLETE AND EXHAUSTIVE LIST OF SASIDHAR'S SKILLS (no other skills exist): {', '.join(full_list)}"


skills_tool = FunctionTool.from_defaults(
	fn=get_skills,
	name="get_skills",
	description="Returns Sasidhar's skills as a list.",
)

agent = ReActAgent(
	tools=[skills_tool],
	llm=llm,
	verbose=True,
	system_prompt=(
		"You are an assistant that STRICTLY provides information about Sasidhar's skills. \n"
		"CRITICAL RULES:\n"
		"1. You MUST call the get_skills tool to retrieve Sasidhar's skills.\n"
		"2. The tool will return a COMPLETE AND EXHAUSTIVE list of Sasidhar's skills.\n"
		"3. You MUST list ONLY and ALL the skills from the tool output - no more, no less.\n"
		"4. Do NOT add, infer, guess, or hallucinate ANY skills not explicitly in the tool output.\n"
		"5. If the user asks about a skill not in the list, respond: 'That skill is not in Sasidhar's skill list.'\n"
		"Answer ONLY based on what the get_skills tool returns. Do not use general knowledge."
	),
)


async def main():
	print("===================================================================")
	print('Querying the ReAct agent: "What are Sasidhar\'s skills?"')
	print("===================================================================")

	response = await agent.run(user_msg="What are Sasidhar's skills?")

	print("\nFinal response:")
	print(response)


if __name__ == "__main__":
	asyncio.run(main())
