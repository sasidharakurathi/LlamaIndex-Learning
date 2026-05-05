<h1 align="center" style="font-size: 60px;">
  LlamaIndex
</h1>

<br>

# MODULE 1 - Core Foundations (LlamaIndex Mental Model)
## Topic 1: LlamaIndex Architecture (Core Mental Model)
-> LlamaIndex is a data orchestration layer bewteen our data and the LLM <br>
-> It solves "How do I efficiently retrieve the right context and feed it to an LLM?"

### The Problem It Solves
#### Generally Raw LLM has following Limitations
-> Have limited context windows <br>
-> Cannot search our database/files natively <br>
-> Most importantly Hallucinate without grounding (without base knowledge)

### The Core Pipeline of LlamaIndex
```text
Raw Data -> Documents -> Nodes -> Index -> Retriever -> Query Engine -> LLM -> Response
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
 - VectorStoreIndex -> It's used for semantic search
 - TreeIndex -> It's used for hierarchical reasoning

**4. Retriever** <br>
Retriever decides **Which nodes are relevant to the query?** <br>
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
- -> We are using `gemini-2.5-flash` as our LLM.

#### Step 2: Embedding Model (Local)
- We are using `HuggingFaceEmbedding` with `BAAI/bge-small-en` model.
    -> It Converts text into a vector (list of numbers) that retriever can compare. <br>
    -> By setting `device="cpu"`, we run the `Embedding Model` on processor / cpu. <br>

#### Step 3: Create Documents
- Wrap our raw strings into LlamaIndex `Document` objects. <br>
    -> This prepares the data to be processed into the data orchestration layer.<br>

#### Step 4: Create Index
- `VectorStoreIndex.from_documents` builds our searchable data structure. <br>
    -> It takes documents -> splits them into Nodes -> generates embeddings for each node -> stores them in memory. <br>

#### Step 5: Query Engine
- Convert the index into a `query_engine`. <br>
    -> This is the **orchestrator** that ties the retrieval (index) and the LLM together. <br>

#### Step 6: Ask
- Send the query: *"What has Sasidhar built?"* <br>
    -> The engine retrieves the most similar nodes and sends them to LLM (Gemini) to extract the final answer. <br>

#### Step 7: Debugging (Retrieval Check)
- Using `index.as_retriever()` to see the raw "retrieval layer". <br>
    -> This allows us to inspect the **Nodes**, their **Similarity Scores** (how relevant the retriever thinks it is), and the **Text content**. <br>
    -> **Crucial for UI/UX:** Helps us verify if the correct context was even found before blaming the LLM for a wrong answer. <br>

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
- Samller chunks -> Better retrieval -> Better LLM performance

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
Documents -> NodeParser -> Nodes -> Embeddings -> Index
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
- **Why?** Overlap ensures that if a sentence is split, the meaning is preserved in both chunks.

#### Step 2: Convert Documents -> Nodes
- We manually execute `node_parser.get_nodes_from_documents(documents)`.
- This transforms our single large `Document` into a list of small `Node` objects.
- This allows us to inspect and verify our chunks before they even get indexed.

#### Step 3: Build Index from Nodes
- Instead of using `from_documents`, we pass the `nodes` list directly into `VectorStoreIndex`.
- Since we already parsed the nodes, LlamaIndex skips the default parsing and uses our custom chunks.
- This is how we take **100% control** over the retrieval unit.

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
Different data -> different chunking:
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
1. VectorStoreIndex -> (default, most used)
2. SummaryIndex -> (global understanding)
3. TreeIndex -> (hierarchical reasoning)

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
Query -> embedding -> vector search -> top-k nodes -> LLM
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
All nodes -> summarized -> LLM answers from summary
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
Leaf nodes -> grouped -> summarized -> higher nodes -> final answer
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
 - VectorIndex -> fetch relevant chunks
 - TreeIndex -> reason over them
 - SummaryIndex -> generate report

This becomes multi-index routing (we will learn it in later module)

#### Troubleshooting Common Index Issues
**1. Why is SummaryIndex giving vague answers?** <br>
Because it relies on a summary representation, it can miss fine-grained details.<br>
**Fix:**
- Use VectorIndex for precise Q&A

**2. TreeIndex is slow** <br>
TreeIndex can be slower due to multiple LLM calls during traversal.
So we need to use it only when necessary (complex reasoning).

**3. VectorIndex misses context**
Because it relies on local similarity, it can miss relevant info if the query is not well-formed or if the embedding model is not good.<br>
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
Documents -> Nodes -> Embeddings -> Index -> Query
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
Nodes -> Stored in -> StorageContext -> Backed by -> Disk / DB / Cloud
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
Query -> Index -> Answer
```
#### *Wrong!!*

### Core Insight
- **Retriever** finds relevant data <br>
- **Query Engine** orchestrates the full answer generation

### Actual Model 
```text
User Query
   ↓
Retriever -> returns nodes (raw data)
   ↓
Query Engine -> uses LLM to synthesize answer
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
 - If yes -> problem is in LLM.
 - If no -> it's a retrieval problem (chunking, embedding, index issue). 

#### Step 2: Check Query Engine
If nodes are correct but answer is wrong, then problem is in prompt / sysntehsis.
 - Check how nodes are being fed into the prompt.
 - Check if LLM is understanding the context.

### Let's see how Controlled Pipleline is practically implemented
#### We now seperate both explicitly
Check the following steps in [main.py](practice/Module%201/Topic%205/main.py).

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
- For retriever -> we optimize chunking, embeddings, index structure.
- For query engine -> we optimize prompts, LLM choice, response synthesis.

<br><hr>

# MODULE 2 - Building Production-Grade RAG
## Topic 6: Embeddings Deep Dive (Gemini, local, hybrid)
### What are Embeddings?
#### We have been using them already:
```python
HuggingFaceEmbedding("BAAI/bge-small-en")
```

#### But let's define it properly:
`Embeddings are numerical representations of text that capture semantic meaning`

#### Core Insight:
`Our RAG system is only as good as our embeddings.`

### How They Work?
#### Example:
```python
"ATS system" -> [0.12, -0.45, 0.88, ...]
"resume parser" -> [0.10, -0.40, 0.85, ...]
```
-> Similar meaning vectors are close in space

### Retrieval Mechanism:
```text
Query -> embedding -> compare with stored vectors -> pick closest
```

#### Distance Metrics:
| Metric       | Description                     | Use Case                |
| ------------ | ------------------------------- | ----------------------- |
| Cosine Similarity | Measures angle between vectors    | Most common for text    |
| Euclidean Distance | Measures straight-line distance   | When magnitude matters  |
| Dot Product   | Measures raw similarity           | Fast but less interpretable |

### Why Embeddings are Critical for RAG?
#### If embeddings are weak:
```text
Query: "projects"
Retrieved: "internships"
```
Entire System fails, even if LLM is perfect

### Types of Embeddings
#### 1. Open Source (Local) <br> Example:
- `BAAI/bge-small-en`
- `all-MiniLM-L6-v`
#### Pros:
- Free to use
- No latency (runs locally)
#### Cons:
 - Lower semantic accuracy compared to top-tier models

 #### 2. Google Gemini Embeddings (Cloud)
#### Pros:
- Better semantic understanding
- Works well with Gemini LLM
#### Cons:
- API usage limits
- Slight latency

### Practical Implementation of Embeddings
#### 1. Our current implementation:
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
    device="cpu"
)
```
#### 2. Improve Retrieval Quality
```python
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en",
    query_instruction="Represent this query for retrieving relevant documents:",
    text_instruction="Represent this document for retrieval:"
)
```
#### Why this is important?
- The BGE models are trained with instructions
- Without instructions, the embeddings are weaker

#### 3. Debugging Embeddings
```python
query = "What projects has Sasidhar built?"

query_embedding = embed_model.get_query_embedding(query)

print(len(query_embedding))  # vector size
print(query_embedding[:5])   # preview
```

### Important issue to consider:
 - Embeddings are not perfect semantics

#### For example:
```text
Query: "strongest skills"
Doc: "backend development, AI systems"
```
 - These might not match strongly

#### Why???
- There is no lexical overlap
- The semantic matching depends on the embedding model

#### Solution:
- Hybrid search (BM25 + vector)
- Reranking with LLM (we'll learn in later modules)
- Query reformulation/rewriting (we'll learn in later modules)

### Nutshell:
```text
Chunking defines WHAT is indexed
Embeddings define HOW it is retrieved
```

### Here is a mini [task](practice/Module%202/Topic%206/task.py) for you to understand embeddings better:
1. Activate python virtual environment
2. Change Directory to [Topic 6](practice/Module%202/Topic%206/)
3. Run `python task.py`
4. You well huge list of floating point numbers in the output. These are the embedding vectors for the queries.
5. Don't get scared by the numbers. Just understand that these vectors are what allow the retriever to find relevant nodes based on semantic similarity. The closer the vectors, the more relevant the nodes are to the query.

## Topic 7: Advanced Chunking Strategies
### Why do we need advanced chunking strategies?
### Let's think in this way `Why Basic Chunking is not enough?`
So far we have used:
```python
SentenceSplitter(chunk_size=150, chunk_overlap=30)
```
This is fixed sixed chunking.
- which means it doesn't consider the content or structure of the document.

### Core point to consider for chunking:
```text
Chunking is NOT just splitting text.
It is defining the semantic boundaries of knowledge.
```

### Problems with Basic Chunking (Fixed Chunk Size)
Example:
```text
"Sasidhar built an ATS system. It includes resume parsing and RAG."
```

With bad split:
```text
Chunk 1: "Sasidhar built an ATS system."
Chunk 2: "It includes resume parsing and RAG."
```

If we ask `What does ATS include?`
 - Either Chunk 2 retrieved (missing ATS context)
 - Or Chunk 1 retrieved (missing features)

Hence, the system / LLM fails to answer correctly.

### Advanced Chunking Strategies
#### There are alot of chunking strategies beyond fixed size. We'll focus on 3 main ones:

| Id | Strategy           | Description                            | Use Case                     |
| -- | ------------------ | -------------------------------------- | ---------------------------- |
| 1  |Section-based      | Split based on document sections        | Resumes, reports             |
| 2  |Semantic chunking  | Split based on semantic meaning         | Web data, unstructured text |
| 3  |Slide window        | Overlapping chunks with a sliding window | Long documents               |

#### 1. Section-based Chunking
It splits documents based on the **Natural Language Boundaries** rather than tokens.<br>
<br>Impemenation:
```python
from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=50
)

nodes = parser.get_nodes_from_documents(documents)
```

- Most commonly we use this for `General Text`, `Resumes`, `Documentations` etc.

Limitations:
- Still not `meaning-aware`

#### 2. Semantic Chunking
It splits documents based on semantic meaning rather than fixed size. It uses an LLM to determine where to split the text based on the content and context.

How it works:
```text
Sentence embeddings -> compare similarity -> split when meaning shifts
```

Implementation:
```python
from llama_index.core.node_parser import SemanticSplitterNodeParser

semantic_parser = SemanticSplitterNodeParser(
    embed_model=embed_model,
    buffer_size=1,        # context window
    breakpoint_percentile_threshold=95
)

nodes = semantic_parser.get_nodes_from_documents(documents)
```
Key Params:
`breakpoint_percentile_threshold`
- Controls split sensitivity

| Value | Behavior     |
| ----- | ------------ |
| 90    | more splits  |
| 95    | balanced     |
| 99    | fewer splits |


- Best for `Web data`, `Mixed-topic text`, `Long documents` etc.

<br>

Why this is powerful?
```text
It preserves meaning -> improves retrieval dramatically
```

#### 3. Sliding Window Chunking
It creates overlapping chunks by sliding a fixed-size window across the text. This ensures that even if a sentence is split, the context is preserved in both chunks.

Example:
```text
Chunk 1: [A B C]
Chunk 2: [B C D]
Chunk 3: [C D E]
```
- Overlapping meaning windows

Implementation:
```python
from llama_index.core.node_parser import SentenceWindowNodeParser

window_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,   # number of sentences per chunk
)

nodes = window_parser.get_nodes_from_documents(documents)
```

- Useful for `Conversational data`, `Logs`, `Q&A systems` and `Context-heavy reasoning` etc.

- This node saves main sentence and saved surrounding sentences in metadata.

### Golden Rule of Chunking:
```text
Different data -> different chunking strategy
```

#### This is how I used different chunking strategies in my projects:
| Use Case            | Best Strategy              |
| ------------------- | -------------------------- |
| ATS / Resume        | SentenceSplitter + overlap |
| AI Assistant memory | Sliding window             |
| Knowledge base      | Semantic chunking          |

### Hybrid Approach (Advanced Pattern)

We can combine: <br>
`Semantic chunking + overlap = Production Grade Chunking`

### Here is a mini [task](practice/Module%202/Topic%207/task.py) for you to understand advanced chunking strategies better:
#### Take a document (like your project description or something like that)
1. **Create 3 versions:**
    - **SentenceSplitter**
    - **SemanticSplitter**
    - **SentenceWindowParser**

2. **For each:**
    - **print nodes**
    - **run SAME query:** <br>
        ```text
        "What projects has Sasidhar built?"
        ```

3. **Compare:**
    - which retrieves best chunks?
    - which gives best answer?

#### Run the task.py(practice/Module%202/Topic%207/task.py) and analyze the output to see how different chunking strategies impact retrieval and final answer quality.

## Topic 8: Retrieval Strategies (Similarity, MMR, Hybrid Search)
### Why Default Retrieval is not enough?
#### So far we have used:
```python
retriever = index.as_retriever()
``` 
#### This defaults to `Top-K Similarity Search` 
 - Basically `Top-K Similarity Search`  retrieves the most similar nodes based on **cosine similarity** of embeddings.
 - Now what is `Cosine Similarity`? 
    - It measures the angle between two vectors. The smaller the angle, the more similar they are.

#### Do not get scared by these terms.
Just think like `Top-K Similarity Search` is like asking "Which nodes are closest to my query in the embedding space?"

#### Default similarity search = high relevance BUT low diversity
 - Now what is relevance and diversity?
    - **Relevance**: how closely the retrieved nodes match the query
    - **Diversity**: how different the retrieved nodes are from each other
 - The `Top-K Similarity Search` often retrieves:
    - similar chunks
    - repeated information
    - misses broader context

### Let us understand the limitations with an example:
#### Query:
```text
"What projects has Sasidhar built?"
```

#### Similarity search (Top-K) might return:
```text
Node 1: ATS project (description)
Node 2: ATS project (features)
Node 3: ATS project (solutions)
```

#### All the nodes are about the same project (ATS).
#### But we want to retrieve `ATS Project` + `Any other projects` Context. (That's how the actual production RAG system should work)

`To solve this problem, we need more advanced retrieval strategies that balance relevance and diversity.`

### Retrieval Strategies:
| Strategy       | Description                            | Use Case                     |
| -------------- | -------------------------------------- | ---------------------------- |
| Similarity Search (Top-K) | Retrieves top K most similar nodes based on `cosine similarity` | General retrieval, when relevance is key |
| Maximal Marginal Relevance (MMR) | Balances relevance and diversity by re-ranking nodes based on both similarity to the query and dissimilarity to already selected nodes | When you want a mix of relevant and diverse information |
| Hybrid Search  | Combines traditional keyword-based search (like `BM25`) with vector similarity search to leverage both lexical and semantic matching | When you have a mix of structured and unstructured data, or when you want to ensure important keywords are included |

#### 1. Similarity Search (Top-K)
How it works?
```text
Query embedding -> find closest vectors -> return top-k
```

Implementation:
```python
retriever = index.as_retriever(
    similarity_top_k=3
)
```

When to use?
- When we want the most relevant nodes
- When we have a good embedding model that captures semantics well

#### 2. Max Marginal Relevance (MMR)
What it does?
- It re-ranks retrieved nodes to balance relevance and diversity.

Re-Ranking Logic:
```text
Score = relevance - redundancy
```

Implementation:
```python
retriever = index.as_retriever(
    similarity_top_k=5,
    vector_store_query_mode="mmr",
    mmr_threshold=0.5
)
```

Here, `mmr_threshold` controls the balance between relevance and diversity.
| Value | Behavior        |
| ----- | --------------- |
| 0.0   | pure diversity  |
| 0.5   | balanced        |
| 1.0   | pure similarity |


#### Why this matters?
Now instead of:
```text
ATS description
ATS features
ATS solutions
```

We might get:
```text
ATS project
Cybersecurity internship
Another project
```

#### You might think why are we getting `Cybersecurity internship` when the query is about projects?`
- This is because in the `Cybersecurity internship` node, there is some mention of `projects` and the MMR algorithm is trying to balance relevance and diversity. So it includes it.

#### When to use MMR?
 - Multi-topic queries
 - portfolio search

#### 3. Hybrid Search (Semantic + Keyword)
Problem with Embeddings:
 - They struggle with exact keyword matching
 - They struggle with rare keywords that are important for retrieval
 - And importantly they struggle with names

**For Example:** <br>
Query:
```text
"ATS system"
```

Embedding may miss:
```text
"Application Tracking System"
```

So, what's the solution??
```text
Combine keyword search (BM25) with vector search (semantic)
```

Implementation:
```python
retriever = index.as_retriever(
    similarity_top_k=3,
    vector_store_query_mode="hybrid"
)
```

Pros and Cons of Hybrid Search:
 -
 - It handles both exacts terms and meaning, improving recall.
 - However, it can be slower and more complex to implement.
 - It requries a proper setup.

### Side by Side Comparison of Retrieval Strategies:
| Strategy   | Strength    | Weakness              |
| ---------- | ----------- | --------------------- |
| Similarity | precise     | redundant             |
| MMR        | diverse     | slightly less precise |
| Hybrid     | most robust | setup complexity      |

### Troubleshooting Common Retrieval Issues
1. Why MMR giving irrelevant nodes?
    - Too much of diversity (low threshold) can lead to irrelevant nodes being included.
    - Fix: Increase `mmr_threshold=0.7` to prioritize relevance more. 
2. Right now `Hybrid Search` is not working, why?
    - Vector store doesn't support keyword search
    - Fix: `ChromaDB` integration. (We will learn in later modules)

### Let's see how to implement these retrieval strategies in practice:
#### Check [task.py](practice/Module%202/Topic%208/task.py) for complete code.
#### We have used [`Top-k similarity retriever`](practice/Module%202/Topic%208/task.py#50) and [`MMR retriever`](practice/Module%202/Topic%208/task.py#51) retrieve the chunks.
#### You can check and compare the output in [`output.txt`](practice/Module%202/Topic%208/output.txt) to see the difference in retrieved nodes and final answer quality between the two strategies.


## Topic 9: Response Synthesis (Refine, Compact, Tree Summarize)
**So, we have learnt strategies to retrieve relevant nodes.** <br>
**But what happens after retrieval?** <br>
**How does the LLM turn multiple nodes into One Answer?** <br>

- Synthesis is the step that takes place after the retrieval of relevant nodes.
- It decided how to combine the retrieved nodes to generate a final answer.

```text
Retrieval decides WHAT is fetched
Synthesis decides HOW it is combined
```
### Why is Synthesis important?
- If we retrieve 5 nodes, how do we combine them into a coherent answer?
- The quality of the final answer depends heavily on the synthesis method used.
- If we use a simple `refine` method, we might get a verbose answer that includes all nodes without proper summarization.
- If we use a `compact` method, we might lose important details in the process of summarization.
- If we use a `tree summarize` method, we can maintain a balance between detail and coherence by summarizing nodes in a hierarchical manner.

### Synthesis Strategies:
| Strategy       | Description                            | Use Case                     |
| -------------- | -------------------------------------- | ---------------------------- |
| **Compact** (Default, Fast)        | Combines all nodes into a single prompt and generates a concise answer | When you want a brief summary that captures the main points |
| **Refine** (Iterative Reasoning)        | Iteratively refines the answer by adding information from each node one by one | When you want a detailed answer that includes all nodes |
| **Tree Summarize** (Hierarchical synthesis)| Builds a hierarchical summary by first summarizing individual nodes and then combining those summaries | When you have a large number of nodes and want to maintain a balance between detail and coherence |

#### 1. Compact Mode (Default)
- It takes all retrieved nodes and combines them into a single prompt for the LLM to generate a concise answer.

Flow:
```text
Nodes -> merged context -> single LLM call -> answer
```

Implementation:
```python
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="compact"
)
```

Pros and Cons:
 - It is super fast and simple to implement.
 - But it can lead to context overload if there are too many nodes.
 - Also it may miss important details in the process of summarization.

#### 2. Refine Mode (Iterative Reasoning)
- It takes each retrieved node one by one and iteratively refines the answer by adding information from each node.

Flow:
```text
Node 1 -> initial answer
Node 2 -> refine answer
Node 3 -> refine again
...
Final answer
```

Implementation:
```python
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="refine"
)
```

Pros and Cons:
- High accuracy and Better reasoning since it considers each node separately.
- It can handle long contexts better than compact mode.
- However, it is slower due to multiple LLM calls and can be more expensive.

#### 3. Tree Summarize Mode (Hierarchical Synthesis)
- It builds a hierarchical summary by first summarizing individual nodes and then combining those summaries.
- Basically, it groups nodes and summarizes them in a tree-like structure.

Flow:
```text
Nodes -> grouped -> summarized -> combined -> final answer
```
Implementation:
```python
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize"
)
```

Pros and Cons:
- It can easily handle large datasets and maintains sturcture outputs.
- It has good usecase for reports, dashboards, and complex reasoning tasks.
- However, it is the slowest and most expensive due to multiple LLM calls and complex processing.

#### Note: `Never use tree summarize for simple queries with few nodes. It is overkill and will just increase latency and cost without improving answer quality.`

### Side by Side Comparison of Synthesis Strategies:
| Mode    | Speed     | Accuracy  | Use Case      |
| ------- | --------- | --------- | ------------- |
| Compact |  Fast    |  Medium | simple Q&A    |
| Refine  |  Medium |  High    | reasoning     |
| Tree    |  Slow |  High    | summarization |

### Note:
 - Most of the time we tune retrieval but forget about systhesis tuning. 
 - But it is equally important to choose the right synthesis strategy based on the use case and the number of retrieved nodes.
 - Wrong synthesis strategy can also lead to poor answer quality even if `retrieval(context)` is perfect.

### When to use which synthesis strategy?
```
Few nodes -> Compact
Many nodes -> Refine
Huge corpus -> Tree
```


### Let's see how to implement these synthesis strategies in practice:
#### Check [task.py](practice/Module%202/Topic%209/task.py) for complete code.
#### Let's analyze and try to understand the [output](practice/Module%202/Topic%209/output.txt) of three Synthesis Strategies.
1. Check line 10 in [output.txt](practice/Module%202/Topic%209/output.txt#L10)
    - This is the default `Compact` mode output. 
    - We can see that the response is a single line answer.
    - And there is no detailed explanation of my projects. It just gives a brief summary.

2. Now check line 21-26 in [output.txt](practice/Module%202/Topic%209/output.txt#L21)
    - This is the `Refine` mode output. 
    - We can see that the response is more detailed and includes information from each node. It iteratively builds the answer by adding details from each retrieved node in a new line.

3. Finally, check line 37-42 in [output.txt](practice/Module%202/Topic%209/output.txt#L37)
    - This is the `Tree Summarize` mode output. 
    - We can see that the response is structured and maintains a hierarchy of information. 
    - It first summarizes individual nodes and then combines those summaries into a final answer.
    - That's the reason the response only contains my `Project Titles` and no explanation. 
    - As we learnt before, the `tree summarize mode` focuses on maintaining structure and hierarchy rather than providing detailed explanations. 
    - It is more suitable for generating structured summaries or reports rather than detailed answers.

#### Response synthesis = prompt engineering layer
```text
LLM prompt = context + instructions
```

We'll learn how to customize this in further modules.

## Topic 10: Evaluation & Debugging RAG (faithfulness, hallucination detection)
### Why Evaluation and Debugging is important?
- Building a RAG system is not just about implementation. It is also about ensuring that the system is working correctly and providing accurate answers.
- Evaluation helps us measure the performance of our RAG system and identify areas for improvement.
- This is the most important step in building production-grade RAG systems. If we don't evaluate and debug properly, we might end up with a system that gives wrong answers or `hallucinations` without us even realizing it.

#### A RAG system has TWO independent failure points:
1. Retrieval failure (wrong context)
2. Generation failure (LLM hallucination)

We must evaluate both separately to identify where the problem lies.

### The 3 Core Metrics We Must Understand:
#### 1. Faithfulness (aka Anti-Hallucination)
- It measures how accurately the generated answer reflects the retrieved context.

#### Example:
**Context:**
```text
"Sasidhar built an ATS system"
```
**Answer:**
```text
"He also built a blockchain system"
```

- This answer is `unfaithful (hallucination)` because it includes information that is not present in the retrieved context. It is a hallucination.

#### 2. Relevance
 - It measures how relevant the retrieved nodes are to the user query.
 - Basically, it tries to answer the question `Are the retrieved nodes actually useful for the query?`

#### Example:
**Query:**
```text
"What projects has Sasidhar built?"
```
**Retrieved:**
```text
"Future improvements..."
```

#### 3. Answer Correctness (End Result)
 - It checks if the final answer generated by the LLM is accurate and complete based on the retrieved context.
 - This evaluation depends on both retrieval and synthesis.

### Evaluation Pipeline:
```text
Query
 ↓
Check retrieved nodes (Relevance)
 ↓
Check answer vs nodes (Faithfulness)
 ↓
Check completeness (Correctness)
```

### Practical Implementation of Evaluation and Debugging
 - Generally there are two approaches to evaluate RAG systems:
    1. Manual Evaluation
    2. Automated Evaluation

#### 1. Manual Evaluation
- It is nothing but us looking at the retrieved nodes and generated answers to see if they are relevant and faithful.
- Actually, this is what we are doing from the beginning. We have been checking the retrieved nodes and generated answers manually to see if they make sense.

#### 2. Automated Evaluation (LlamaIndex Built-in)
 - Faithfulness Evaluator
    ```python
    from llama_index.core.evaluation import FaithfulnessEvaluator

    evaluator = FaithfulnessEvaluator(llm=llm)

    result = evaluator.evaluate_response(
        response=response_text,
        contexts=[n.text for n in nodes]
    )

    print(result.passing)
    print(result.score)
    ```
 - Relevance Evaluator
    ```python
    from llama_index.core.evaluation import RelevancyEvaluator

    evaluator = RelevancyEvaluator(llm=llm)

    result = evaluator.evaluate_response(
        query=query,
        response=response_text,
        contexts=[n.text for n in nodes]
    )

    print(result.score)
    ```

### What does these scores mean?
| Score   | Meaning           |
| ------- | ----------------- |
| 0.8 - 1.0 | good              |
| 0.5 - 0.8 | needs improvement |
| < 0.5    | broken            |

### Note:
#### Most of the failures are NOT LLM failures.
```text
80% issues = retrieval
20% issues = generation
```

### Debug Strategy
| Problem       | Fix                         |
| ------------- | --------------------------- |
| Wrong nodes   | embeddings / chunking / MMR |
| Hallucination | prompt / refine mode        |
| Missing info  | increase top_k              |

#### Sometimes `Evaluator may say bad but answer looks fine`
 - This is a classic evaluator LLM limitation
 - Always do a `manual check` to confirm if the evaluation is correct or not.

#### Let's see how to implement evaluation and debugging in practice:
 - Check [task.py](practice/Module%202/Topic%2010/task.py) for complete code.
 - In this task, we have implemented both `FaithfulnessEvaluator` and `RelevancyEvaluator` to evaluate the generated answer based on the retrieved nodes and the user query.
 - We have printed the scores for both faithfulness and relevance in the output. 
 - You can analyze these scores to understand how well the RAG system is performing in terms of retrieving relevant nodes and generating faithful answers.
 - Check the output in [output.txt](practice/Module%202/Topic%2010/output.txt) to see the evaluation results for both metrics. 
 - You can also compare these scores with the actual retrieved nodes and generated answer to see if they align with your manual evaluation.

#### So we have come to the end of Module 2. I hope you have understood the concepts. And my notes is clear to you. 😊
#### In the next module, we shift into real-world system design.

# MODULE 3 - Data Connectors & Pipelines
## Topic 11. Data Connectors (PDFs, APIs, DBs, Notion, etc.)
### Why Data Connectors are important?
 - So far we have been working with simple text documents.
 - But in real-time applications, we need to ingest data from various sources like PDFs, APIs, databases, Notion, etc.
    ```text
    PDFs / APIs / DB / Notion / Files → Documents → Nodes → Index
    ```
 - Our RAG system is only as good as our ingestion pipeline.

### What are Data Connectors?
 - Data connectors are tools that allow us to connect to different data sources and ingest that data into our LammaIndex Documents.

#### Examples:
| Source     | Connector             |
| ---------- | --------------------- |
| PDF        | PDFReader             |
| Text files | SimpleDirectoryReader |
| Web pages  | Web loaders           |
| APIs       | Custom loaders        |
| Databases  | SQL loaders           |

### Internal Flow of Data Connectors:
```text
External Data → Loader → Document → NodeParser → Nodes
```

### Practical Implementation of Data Connectors
#### 1. Local Directory Loader:
#### Implementation:
```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True
).load_data()
```

- This reads all files in the `./data` directory and converts them into Documents automatically.

#### 2. PDF Loader
#### Implementation:
```python
from llama_index.readers.file import PDFReader

loader = PDFReader()
documents = loader.load_data(file="resume.pdf")
```

- This reads a PDF file and converts it into a Document.
- PDFs are tricky because they have complex layouts
- so using a specialized PDF reader is important to extract text correctly.
- We'll learn about more advanced PDF parsing techniques in later modules.

#### 3. Custom Loader
- This is where our acutal engineer skills come into play.
- We can build custom loaders to connect to APIs, databases, or any other data source.

#### Example: Load from Database
```python
from llama_index.core import Document

def load_from_db():
    rows = [
        {"name": "Sasidhar", "project": "ATS system"},
        {"name": "Sasidhar", "project": "AI assistant"}
    ]

    documents = []

    for row in rows:
        text = f"{row['name']} built {row['project']}"
        documents.append(Document(text=text))

    return documents
```

#### Example: Load from API
```python
import requests
from llama_index.core import Document

def load_from_api():
    data = requests.get("https://api.example.com/projects").json()

    docs = []

    for item in data:
        docs.append(Document(text=item["description"]))

    return docs
```

### Core Consideration:
- **Documents are NOT just text**
- They can include metadata.
```
Document(
    text="ATS system...",
    metadata={
        "type": "project",
        "source": "portfolio",
        "year": 2025
    }
)
```
- This metadata can be used for filtering, routing and structured retrieval, which we will learn in later modules.

### Here is a mini [task](practice/Module%203/Topic%2011/task.py) for you to understand data connectors better:
1. Create a folder:
    ```bash
    data/
    ├── ats.txt
    ├── another_project.txt
    ```
2. Load using:
    ```python
    SimpleDirectoryReader
    ```
3. Build Index and Query:
    ```text
    "What projects has Sasidhar built?"
    ```

#### Observe the [task.py](practice/Module%203/Topic%2011/task.py), [data/](practice/Module%203/Topic%2011/data/) and [output.txt](practice/Module%203/Topic%2011/output.txt) to see how the data connectors work in practice. 

## Topic 12. Ingestion Pipelines (Transformations, metadata enrichment)
#### Right now we are just injecting raw text into our RAG system.
```text
Loader → Documents → NodeParser → Index
```

#### Basically `Raw data → bad chunks → bad embeddings → bad retrieval`

So we must transform data before indexing.

### What is a Ingestion Pipeline?
- An ingestion pipeline is a series of steps that process and transform raw data into a format that can be indexed and retrieved effectively by our RAG system.
- This is a critical step in building production-grade RAG systems because the quality of our data directly impacts the quality of our retrieval and generation.
- Generally we do not get a perfect cleaned data from the organizations or anyother comercial data sources.
- We need to build pipelines that can handle data cleaning, transformation, and enrichment before it gets indexed.

Here is the updated flow of our RAG system with ingestion pipelines:
```text
Loader → Transformations → Clean Data → NodeParser → Nodes → Index
```

### Types of Transformations:
We will focus on 3 critical transformations:
1. Cleaning (remove noise)
2. Metadata enrichment
3. Custom chunking logic

#### 1. Cleaning
Right now my documents have a lot of noise (markdown syntaxt) like:
```text
**Github Link**
**Demo Link**
**Status**
```
**This might messup the embeddings. So, we need to clean it.**

#### Implementation:
```python
def clean_text(text: str) -> str:
    lines = text.split("\n")
    
    filtered = [
        line for line in lines
        if "Github Link" not in line
        and "Demo Link" not in line
        and line.strip() != ""
    ]
    
    return "\n".join(filtered)
```
#### Apply cleaning in the pipeline:
```python
cleaned_text = clean_text(knowledge_base)

documents = [Document(text=cleaned_text)]
```

#### 2. Metadata Enrichment
- This is the game changer for building production-grade RAG systems.
- We can metadata to our documents that can be used for filtering and structured retrieval later on.

```python
Document(
    text="ATS system...",
    metadata={
        "type": "project",
        "category": "AI",
        "source": "portfolio"
    }
)
```

Later we can:
```python
Retrieve only "projects"
Filter by "AI"
Route queries
```

#### Example
```python
documents = [
    Document(
        text=knowledge_base,
        metadata={"type": "project", "name": "ATS"}
    )
]
```

#### 3. Custom Transformation Pipeline
- We can build custom transformation pipelines that include multiple steps like cleaning, enrichment etc.

```python
def build_documents(raw_text):
    cleaned = clean_text(raw_text)
    
    return [
        Document(
            text=cleaned,
            metadata={"type": "project"}
        )
    ]
```

- Now are pipeline looks like:
```python
documents = build_documents(knowledge_base)

nodes = parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes, embed_model=embed_model)
```

### LammaIndex provides Native Pipeline Support
This is a basic transformations
```python
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        node_parser,
    ]
)

nodes = pipeline.run(documents=documents)
```

Later we can add `cleaners`, `metadata injectors` and `embeddings`.

Note:
 - Always remember, Better preprocessing leads to better retrieval (it's better than changing LLM)
 - Mostly, Metadata enrichment is better than Prompt Engineering. Because `filtering is easier and accurate than guessing`
    - Here filtering can be done based on metadata
    - And guessing is when we try to include instructions in the prompt to make the LLM understand what to do. But it is not always effective.

### Here is a mini [task](practice/Module%203/Topic%2012/task.py) for us to understand ingestion pipelines better:
1. Let's try to clean our markdown documents
 - remove links, empty lines
2. And then add metadata to our documents
    ```python
    metadata={
        "type": "project",
        "name": "ATS",
        "domain": "AI"
    }
    ```
3. Then build the index and query:
    ```text
    "What projects has Sasidhar built?"
    ```
4. Then let's print nodes to verfiy whether:
 - The chunks are clean and not noisy
 - Better relevance due to metadata enrichment

#### Actually, here I am not implementing the metadata filtering. we will learn about that in later modules. But you can see how the metadata is added to the documents and how it can be used for filtering later on.

#### Check the [task.py](practice/Module%203/Topic%2012/task.py) and [output.txt](practice/Module%203/Topic%2012/output.txt) to see how the ingestion pipeline works in practice.
#### Also check the [cleaned_knowledge_base.txt](practice/Module%203/Topic%2012/cleaned_knowledge_base.txt) to see how the raw knowledge base is transformed into a clean and enriched format before indexing.

#### The major cleaning logic is at line 19 in [task.py](practice/Module%203/Topic%2012/task.py#19).
```python
cleaned_document = document.replace("#", "").replace("*", "").strip()
```
Note:
 - Cleaning logic varies based on the type of noise in your data. You can customize it as per your needs.
 - We will cover that in later modules.