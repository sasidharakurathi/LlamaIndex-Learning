# LlamaIndex

## Topic 1: LlamaIndex Architecture (Core Mental Model)
→ LlamaIndex is a data orchestration layer bewteen our data and the LLM <br>
→ It solves "How do I efficiently retrieve the right context and feed it to an LLM?"

### The Problem It Solves
#### Generally Raw LLM has following Limitations
→ Have limited context windows
→ Cannot search our database/files natively
→ Most importantly Hallucinate without grounding (without base knowledge)

### The Core Pipeline of LlamaIndex
```text
Raw Data → Documents → Nodes → Index → Retriever → Query Engine → LLM → Response
```
<br>

| Layer               | Role              |
| ------------------- | ----------------- |
| Data Layer          | Documents / Nodes |
| Storage Layer       | Index             |
| Retrieval Layer     | Retriever         |
| Orchestration Layer | Query Engine      |
| Intelligence Layer  | LLM               |

<br>

**1. Documents**
 - Raw inputs (PDFs, DB rows, APIs, text files etc)
    ```python
    Document(text="Sasidhar is a backend AI engineer...")
    ```
**2. Nodes (Imp concept)** <br>
Documents are split into Nodes
A Node is an collection of
 - chunk of text
 - metadata
 - embedding

Why **chunk of text**??
 - LLMs can't process huge text (due to limited input tokens)
 - So the retrieval of data works at chunk level rather than document level

**3. Index** <br>
An Index is a data structure which represents the nodes in a sturcutred way for retrieval. <br>
In simple words, **Index** = how data is organized for retrieval. <br>

Example:
 - VectorStoreIndex → It's used semantic search
 - TreeIndex → It's used for hierarchical reasoning

**4. Retriever** <br>
Retriever decided **Which nodes are relevant to the query?** <br>
Bascially, it does following things:
 - Similarity search
 - Filtering
 - Ranking

**5. Query Engine** <br>
Query Engine is an orchestrator.
1. It takes the user query.
2. Then Calls retriever.
3. Gets relevant nodes
4. Sends those nodes to LLM.
5. Synthesizes the response from LLM
6. Returns the final answer to user.

#### This is the core step in production based RAG system. <br>

**6. LLM** <br>
LLM generates the final answer based on the **user query** and the **retrieved context** (aka nodes). <br>
Note: In this complete learning roadmap, we will be using Gemini as our LLM. <br>

### Let's see how it works via practical implementation
Let's try to build the smallest correct architecture of LlamaIndex. <br>

1. Make sure to create an Python environment. (This course uses Python 3.11.9)
2. Then inside this repo, go to [/practice/Topic 1](practice/Topic%201/)
3. Open [.env.example](practice/Topic%201/.env.example) and replace `YOUR_GEMINI_API_KEY` with your gemini api key
4. Rename `.env.example` to `.env`
5. Install all requriements
    ```bash
    pip install -r requriements.txt
    ```
6. Execute the `main.py`
    ```bash
    python main.py
    ```

Here is the code flow and output explanation

#### Step 0: Load and setup GEMINI_API_KEY
- We are loading the `.env` file using `python-dotenv`

#### Step 1: Configure LLM (Gemini)
- Initialize the `GoogleGenAI` model.
- → We are using `gemini-2.5-flash` as our LLM.

#### Step 2: Embedding Model (Local)
- We are using `HuggingFaceEmbedding` with `BAAI/bge-small-en` model.
- → It Converts text into a vector (list of numbers) that retriever can compare.
- → By setting `device="cpu"`, we run the `Embedding Model` on processor / cpu.

#### Step 3: Create Documents
- Wrap our raw strings into LlamaIndex `Document` objects.
- → This prepares the data to be processed into the data orchestration layer.

#### Step 4: Create Index
- `VectorStoreIndex.from_documents` builds our searchable data structure.
- → It takes documents → splits them into Nodes → generates embeddings for each node → stores them in memory.

#### Step 5: Query Engine
- Convert the index into a `query_engine`.
- → This is the **orchestrator** that ties the retrieval (index) and the LLM together.

#### Step 6: Ask
- Send the query: *"What has Sasidhar built?"*
- → The engine retrieves the most similar nodes and sends them to LLM (Gemini) to extract the final answer.

#### Step 7: Debugging (Retrieval Check)
- Using `index.as_retriever()` to see the raw "retrieval layer".
- → This allows us to inspect the **Nodes**, their **Similarity Scores** (how relevant the retriever thinks it is), and the **Text content**.
- → **Crucial for UI/UX:** Helps us verify if the correct context was even found before blaming the LLM for a wrong answer.

#### Step 8: Output (Example)
```text

Response(response='Sasidhar has built an ATS system that includes resume '
                  'parsing and an LLM.',...)

--- Node 1 ---
Score: 0.878144731169124
Text: Sasidhar is an AI engineer specializing in backend systems.

--- Node 2 ---
Score: 0.7303317940960742
Text: He built an ATS system with resume parsing and LLM.
```