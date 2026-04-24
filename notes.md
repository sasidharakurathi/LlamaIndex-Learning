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
2. Then inside this repo, go to [Topic 1](practice/Module%201/Topic%201/)
3. Open [.env.example](practice/Module%201/Topic%201/.env.example) and replace `YOUR_GEMINI_API_KEY` with your gemini api key
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

## Topic 2: Documents vs Nodes vs Chunking (Where Most Systems Fail)

### 1. Documents (Input Layer - Temporary)
```python
Document(text="Sasidhar built an ATS system...")
```
- This is the raw input layer.
- It can be anything (PDF, text, DB rows, API response etc).
- These existing only during ingestion
- They are not used for retrieval or LLM input directly.

In simple terms **Documents are just a source to generate Nodes.**

### 2. Nodes (The Real Unit of Intelligence)
As we discussed before, a node is a collection of **chunk of text** + **metadata** + **embedding**.

Example representation of a Node:
```text
Node:
- text: "Sasidhar built an ATS system..."
- metadata: {}
- embedding: [vector]
```

### Why Nodes Exist?
Because LLMs can't process huge text (due to limited input tokens), we need to break down documents into smaller chunks of text (nodes) for efficient retrieval and feeding into the LLM.
- Samller chunks → Better retrieval → Better LLM performance

### 3. Chunking (THE MOST IMPORTANT STEP IN RAG)
Chunking is the process of breaking down documents into smaller, manageable pieces (nodes). <br>
Chunking decides whether your system is smart or useless. <br>

#### Example of Bad Chunking:
```text
[Whole resume in one chunk] or
[Too small chunks like single sentences or single words]
```
#### Problems:
- Poor retrieval
- Irrelevant context
- Hallucinations

#### Example of Good Chunking:
```text
"Sasidhar built an ATS system with resume parsing and RAG."
```
#### Benefits:
- Self-contained meaning
- Context preserved
- Embedding is meaningful

### What LlamaIndex Does by Default
When we run:
```python
VectorStoreIndex.from_documents(documents)
```
It internally:
```text
Documents → NodeParser → Nodes → Embeddings → Index
```

### Let's see how we can control our chunking via practical implementation
Now we take control. No more defaults.

1. Go to [Topic 2](practice/Module%201/Topic%202/)
2. Open [.env.example](practice/Module%201/Topic%202/.env.example) and replace `YOUR_GEMINI_API_KEY` with your gemini api key
3. Rename `.env.example` to `.env`
4. Install all requriements
    ```bash
    pip install -r requriements.txt
    ```
5. Execute the `main.py`
    ```bash
    python main.py
    ```

Here is the code flow and output explanation

#### Step 1: Use a Node Parser (Explicit Chunking)
- We use `SentenceSplitter` to define exactly how our text should be broken down.
- `chunk_size=100`: This limits each chunk to 100 tokens.
- `chunk_overlap=20`: This keeps 20 tokens of context from the previous chunk.
- → **Why?** Overlap ensures that if a sentence is split, the meaning is preserved in both chunks.

#### Step 2: Convert Documents → Nodes
- We manually execute `node_parser.get_nodes_from_documents(documents)`.
- → This transforms our single large `Document` into a list of small `Node` objects.
- → This allows us to inspect and verify our chunks before they even get indexed.

#### Step 3: Build Index from Nodes
- Instead of using `from_documents`, we pass the `nodes` list directly into `VectorStoreIndex`.
- → Since we already parsed the nodes, LlamaIndex skips the default parsing and uses our custom chunks.
- → This is how we take **100% control** over the retrieval unit.

#### Step 4: Output (Example)
```text
Loading weights: 100%|████████████████████████████| 199/199 [00:00<00:00, 11383.89it/s]

--- Node 1 ---
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem...

--- Node 2 ---
It was popularised in the 1960s with the release of Letraset sheets containing...

--- Node 3 ---
The point of using Lorem Ipsum is that it has a more-or-less normal distribution...

--- Node 4 ---
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots...

--- Node 5 ---
Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et...

--- Node 6 ---
There are many variations of passages of Lorem Ipsum available, but the majority...

--- Node 7 ---
It uses a dictionary of over 200 Latin words, combined with a handful of model...
```

#### Trobleshooting Common Chunking Issues
**1. My answers are irrelevant** <br>
90% of the time, it's a chunking issue. Check if the nodes are meaningful and self-contained.<br>
**Fix:**
- Increase chunk size
- Add more overlap

**2. Retriever misses important info** <br>
If your chunks are too small or too large, the retriever might miss the relevant context.
**Fix:**
- Increase overlap
- Use semantic chunking (later topic)

**3. Too many irrelevant nodes retrieved**
If our chunks are too large, they might contain a lot of irrelevant information, leading to noisy retrieval.
**Fix:**
- Decrease chunk size

### Advanced Insight (Chucking Strategies)
Different data → different chunking:
| Data Type | Strategy                |
| --------- | ----------------------- |
| Resume    | section-based chunking  |
| Code      | function-level chunking |
| Docs      | paragraph-based         |
| Chat logs | conversation windows    |
| Web data  | semantic chunking       |
| Markdown  | heading-based chunking  |
| FAQs      | Q&A pair chunking       |

#### We will build custom chunkers later in the learning roadmap.

### Here is a mini [task](practice/Module%201/Topic%202/task.py) for you to understand chunking better:
Take the previous project from Topic 1 and: <br>
**1. Replace:**
```python
VectorStoreIndex.from_documents(...)
```
**With:**
- SentenceSplitter
- Manual node creation

**2. Experiment with:**
```python
chunk_size = 50
chunk_size = 150
chunk_size = 300
```
**3. Observe how the retrieved nodes and final answer changes with different chunking strategies.**

**Note: Make sure to use long documents like your resume or a long article to see the impact of chunking.**

## Topic 3: Index Types Deep Dive (Vector vs Summary vs Tree)
### What is an Index??

We have already used:
```python
VectorStoreIndex
```
But what is it really? <br>
An Index = a strategy for organizing nodes so they can be retrieved efficiently for a query. <br>

### Core point:
```text
Retriever quality ≠ just embeddings
Retriever quality = Index + Embeddings + Chunking
```

### The 3 Core Index Types

We'll focus on:
1. VectorStoreIndex → (default, most used)
2. SummaryIndex → (global understanding)
3. TreeIndex → (hierarchical reasoning)

| Index Type       | Use Case                          | Retrieval Method          |
| ---------------- | --------------------------------- | ------------------------- |
| VectorStoreIndex | Semantic search (similarity)      | Vector similarity search  |
| SummaryIndex     | Retrieval via summarization       | LLM-based summarization   |
| TreeIndex        | Hierarchical reasoning and retrieval | Tree traversal + LLM reasoning |

### 1. VectorStoreIndex (Default Index)
#### What id does
- It stores embeddings of nodes in a vector database (in-memory or external).
- And it uses similarity search to retrieve relevant nodes based on the query embedding.

#### How VectorStoreIndex works internally
```text
Query → embedding → vector search → top-k nodes → LLM
```

#### When to use VectorStoreIndex?
- Q&A systems
- Search engines
- RAG pipelines
- ATS / document retrieval

#### Cons of VectorStoreIndex
- No global understanding, it only retrieves based on local similarity.
- Purely similarity-based, so it can miss relevant info if the query is not well-formed or if the embedding model is not good.

#### Implementation of VectorStoreIndex
- We have already implemented it in Topic 1 and 2.
    ```python
    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )
    ```

- Advanced Control
    ```python
    query_engine = index.as_query_engine(
        similarity_top_k=3  # control retrieval depth
    )
    ```
- This allows us to retrieve the top 3 most similar nodes for any query.

### 2. SummaryIndex (Underrated but Powerful)
#### What id does
- It builds a summary representation of all nodes using an LLM.
- During retrieval, it uses the LLM to match the query against the summary instead of individual nodes.
- This allows it to have a more global understanding of the content, rather than relying solely on local similarity.
#### How SummaryIndex works internally
```text
All nodes → summarized → LLM answers from summary
```

#### When to use SummaryIndex?
- Give me an overview
- Summarize this document
- Reports, dashboards

#### Cons of SummaryIndex
- Loses fine-grained details since it relies on a summary representation.
- Can be less accurate for specific queries that require detailed information.
- Nod good for precie Q&A or retrieval tasks that need exact matches.

#### Implementation of SummaryIndex
```python
from llama_index.core import SummaryIndex

summary_index = SummaryIndex(nodes)

query_engine = summary_index.as_query_engine(llm=llm)

response = query_engine.query("Summarize Sasidhar's experience")
print(response)
```

### 3. TreeIndex (Hierarchical Reasoning)
#### What id does
- It builds a tree structure of nodes based on their relationships (e.g., parent-child).
- Each level summarizes the nodes below it, creating a hierarchy of information.
- During retrieval, it traverses the tree to find relevant nodes and can perform reasoning across the hierarchy.

#### How TreeIndex works internally
```text
Leaf nodes → grouped → summarized → higher nodes → final answer
```

#### When to use TreeIndex?
- Large documents
- Complex reasoning
- Multi-step summarization

#### Cons of TreeIndex
- More complex to build and maintain.
- Can be slower for retrieval due to tree traversal and multiple LLM calls.
- More expensive (more LLM calls).

#### Implementation of TreeIndex
```python
from llama_index.core import TreeIndex

tree_index = TreeIndex(nodes, llm=llm)

query_engine = tree_index.as_query_engine(llm=llm)

response = query_engine.query("Explain Sasidhar's overall profile")
print(response)
```

### Side by Side Comparison of Index Types
| Feature              | Vector | Summary  | Tree           |
| -------------------- | ------ | -------- | -------------- |
| Retrieval            | ✅ Yes  | ❌ No     | ⚠️ Indirect    |
| Global understanding | ❌ No   | ✅ Yes    | ✅ Yes              |
| Precision            | ✅ High | ❌ Low    | ✅ Medium       |
| Speed                | ⚡ Fast | ⚡ Fast   | 🐢 Slow        |
| Use case             | RAG    | Overview | Deep reasoning |

### Implementation of Index in Real time systems:
#### In real time systems, we never rely on just one index type. We often combine them for better performance.

Example:
 - VectorIndex → fetch relevant chunks
 - TreeIndex → reason over them
 - SummaryIndex → generate report

This becomes multi-index routing (we will learn it in later module)

#### Troubleshooting Common Index Issues
**1. Why is SummaryIndex giving vague answers?** <br>
Because it relies on a summary representation, it can miss fine-grained details.
**Fix:**
- Use VectorIndex for precise Q&A

**2. TreeIndex is slow** <br>
TreeIndex can be slower due to multiple LLM calls during traversal.
So we need to use it only when necessary (complex reasoning).

**3. VectorIndex misses context**
Because it relies on local similarity, it can miss relevant info if the query is not well-formed or if the embedding model is not good.
**Fix:** (We will learn these techniques in later modules)
- hybrid retrieval
- reranking
- recursive retrieval

### Here is a mini [task](practice/Module%201/Topic%203/task.py) for you to understand index:
Take the previous project from Topic 1 and modfiy it to use `SummaryIndex` instead of `VectorStoreIndex`. <br>
Check how the retrieved nodes and final answer changes with this new index type. <br>
Then, try to implement `TreeIndex` and compare the results of all three index types side by side.

## Topic 4: StorageContext & Persistence (Stateful LlamaIndex Systems)
Right now, every time we run our script, embeddings are generated from scratch, and the index is built from scratch. This increases startup latency<br>
```text
Documents → Nodes → Embeddings → Index → Query
```
In real-world applications, we need to **persist** our index so that we can reuse it across multiple runs and maintain state. <br>
Without persistence, our system is NOT production-ready

### What is StorageContext?
StorageContext is an abstraction layer in LlamaIndex that manages where our index data lives and how it's stored. It allows us to plug in different storage backends (in-memory, disk, cloud) without changing our core logic.

It handles:
 - Vector store (embeddings)
 - Document store
 - Index metadata

Mind Model:
```text
Nodes → Stored in → StorageContext → Backed by → Disk / DB / Cloud
```

### Components Inside StorageContext
| Component    | Role                   |
| ------------ | ---------------------- |
| docstore     | stores nodes           |
| vector_store | stores embeddings      |
| index_store  | stores index structure |

### Let's see the practical implementation of Local Persistence

We'll use:
- Local disk (free)
- Default storage (simple and clean)

1. Inside this repo, go to [Topic 4](practice/Module%201/Topic%204/)
2. Open [.env.example](practice/Module%201/Topic%204/.env.example) and replace `YOUR_GEMINI_API_KEY` with your gemini api key
3. Rename `.env.example` to `.env`
4. Install all requriements
    ```bash
    pip install -r requriements.txt
    ```
5. Execute the `main.py`
    ```bash
    python main.py
    ```

#### Step 1: Create StorageContext
```python
index.storage_context.persist(persist_dir=PERSIST_DIR)
```

#### Step 2: Load StorageContext
```python
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage

storage_context = StorageContext.from_defaults(
    persist_dir="./storage"
)

index = load_index_from_storage(storage_context)
```
#### Check [main.py](practice/Module%201/Topic%204/main.py) for complete code.

#### This is the production design pattern
Now:
 - No re-embedding
 - No re-indexing
 - Instant startup

#### Troubleshooting Common Persistence Issues
**1. Why My changes aren't reflected?** <br>
Because we are loading old persisted index.
**Fix:**
 - Make sure to clear the storage directory if you want to start fresh.
    ```bash
    rm -rf storage/ # Linux/Mac
    rd /s /q storage # Windows
    ```
 - Then rebuild the index and persist it again.

**2. Embedding model mismatch** <br>
If we change the embedding model but reuse the old persisted index, the retrieval will be poor because the embeddings won't match the new model's vector space.<br>
And the results will break silently without any error.

**Note:** `Same embedding model MUST be used for loading`

**3. Corrupted storage / weird errors**
This happens if the storage files get corrupted or if there are permission issues or if version mismatch of LlamaIndex.
**Fix:** (We will learn these techniques in later modules)
- delete storage
- rebuild

### Advanced Insight (Other types of storage)
#### In real systems, we don't just use disk:
| Storage             | Use case         |
| ------------------- | ---------------- |
| Local disk          | dev / small apps |
| ChromaDB            | local + scalable |
| Pinecone / Weaviate | production cloud |

#### We'll learn ChromaDB in next modules.

## Topic 5: Query Engine vs Retriever (Critical Distinction)

### What's the Confusion?
#### Most fo the devs think:
```text
Query → Index → Answer
```
#### *Wrong!!*

### Core Insight
- **Retriever** finds relevant data <br>
- **Query Engine** orchestrates the full answer generation

### Actual Model 
```text
User Query
   ↓
Retriever → returns nodes (raw data)
   ↓
Query Engine → uses LLM to synthesize answer
```

### 1. Retriever (Data Fetching Layer)
#### What it does
- It takes the user query
- And retrieves relevant nodes from the index.
- Returns them

That's it. No LLM involved. No answer generation. Just data fetching.

#### Example:
```python
retriever = index.as_retriever(similarity_top_k=3)

nodes = retriever.retrieve("What has Sasidhar built?")

for node in nodes:
    print(node.text)
```

#### Output:
```text
"Sasidhar built an ATS system..."
"He developed a backend AI system..."
```

### 2. Query Engine (Orchestration Layer)
#### What it does
- It calls retriever to get relevant nodes
- Sends those nodes to the LLM along with the user query
- Generates the final answer by synthesizing the retrieved nodes and the query.

#### Example:
```python
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What has Sasidhar built?")
print(response)
```

#### Internal Flow:
```text
1. retrieve nodes
2. build prompt:
   "Context: <nodes> + Query"
3. send to LLM
4. synthesize response
```

### Side-by-Side Comparison
| Feature  | Retriever            | Query Engine     |
| -------- | -------------------- | ---------------- |
| Uses LLM | ❌ No                 | ✅ Yes            |
| Output   | Nodes                | Final answer     |
| Control  | High                 | Abstracted       |
| Use case | Debugging, pipelines | End-user queries |


### Why is this distinction important?
 - If our system fails:
    ```text
    We MUST isolate:
    Is it retrieval problem OR LLM problem?
    ```

### Debugging Strategy:
#### Step 1: Check Retriever
```python
nodes = retriever.retrieve(query)
```
Are nodes correct and relevant? 
 - If yes → problem is in LLM.
 - If no → it's a retrieval problem (chunking, embedding, index issue). 

#### Step 2: Check Query Engine
If nodes are correct but answer is wrong, then problem is in prompt / sysntehsis.
 - Check how nodes are being fed into the prompt.
 - Check if LLM is understanding the context.

### Let's see how Controlled Pipleline is practically implemented
#### We now seperate both explicitly
Check the following steps in [main.py](practice/Module%201/Topic%205/main.py).

(practice/Module%201/Topic%205/main.py#L50)

#### Step 1: Retriever: [main.py line 50-51](practice/Module%201/Topic%205/main.py#L50)
 - We create a retriever from the index using `index.as_retriever()`.
 - This gives us direct access to the retrieval layer, allowing us to fetch nodes without involving the LLM.

#### Step 2: Manual Pipeline (Important): [main.py line 53-68](practice/Module%201/Topic%205/main.py#L53)
 - We manually call `retriever.retrieve(query)` to get the relevant nodes.
 - Then we build a custom prompt by combining the retrieved nodes with the user query.
 - Finally, we send this prompt to the LLM to get the final answer.

### Why This is Powerful?
#### Now we can:
 - customize prompts
 - add reranking
 - inject memory
 - control everything
#### This is how real production systems are built. We don't rely on black-box query engines. We have full control over the retrieval and synthesis process.

### Troubleshooting Common Query Engine Issues
**1. “Answer is wrong but nodes are correct** <br>
This means the retriever is working but the LLM is not synthesizing the answer correctly.
**Fix:**
 - better prompt engineering
 - use a more powerful LLM
 - response mode (we'll learn in next modules)

**2. Retriever returns irrelevant nodes** <br>
Cause:
 - embeddings
 - chunking
 - top_k too small

**Fix:**
- improve chunking
- use better embedding model
- increase `similarity_top_k` to fetch more nodes

**3. Query engine hides too much**
Yes, query engines are great for simplicity, but they can hide the retrieval process, making it harder to debug when things go wrong. <br>

For Production:
- we often bypass it and build custom pipelines for better control and observability.

### Advanced Insight
```text
Retriever = Recall problem
Query Engine = Reasoning problem
```

#### We optimize them differently:
- For retriever → we optimize chunking, embeddings, index structure.
- For query engine → we optimize prompts, LLM choice, response synthesis.