# PoC Demo Summary & Narrative

This document outlines the story and key talking points for the proof-of-concept demo. The central theme is a direct comparison between a **"Classic RAG"** system (which we use today) and a **"Hybrid Graph RAG"** system to prove the business value of incorporating a knowledge graph.

### Core Question from Leadership:
*   **"How do we assess that this is better than our classic RAG platform?"**
*   **"Can we use a more cost-effective LLM, focusing on performance-per-dollar?"**

Our entire demo is structured to answer these questions.

---

## Architecture and Technology Choices

*   **Language Model for Graph Building**: We will use **OpenAI's `gpt-4o-mini`**. This is a deliberate choice to test a new, highly cost-effective model that is optimized for performance-per-dollar, directly addressing leadership feedback.
*   **Vector Database**: **Pinecone** (Represents the "Classic RAG" semantic search capability).
*   **Graph Database**: **Neo4j Aura** (Represents the structured knowledge, filtering, and relationship analysis layer).
*   **Graph Builder Engine**: **LangChain's `LLMGraphTransformer`**. This library is used to orchestrate the `gpt-4o-mini` model to extract structured entities (nodes, relationships) from unstructured text. It is representative of modern automated KG construction tools.
*   **Application Framework**: **Streamlit** (For rapid, interactive demo development).
*   **Data Sources**:
    *   **Semantic Scholar API**: Provides real, verifiable academic papers.
    *   **Simulated Grant Data (`data/grants_al.json`)**: A curated dataset designed to create a clear, repeatable, and easily understandable filtering scenario. We explicitly state this is simulated to maintain credibility.

---

## The Demo: A Three-Act Story

### Act 1: The LLM Builder - Can we trust it?

**Goal**: Prove that a cost-effective LLM like `gpt-4o-mini` can reliably extract structured facts from unstructured text. This builds the foundation of trust for everything that follows.

**How it Works**:
1.  We take the abstract of a well-known research paper.
2.  We show the raw text and its real, verifiable citation count from the source (Semantic Scholar).
3.  We feed *only the text* to `gpt-4o-mini` via our ingestion script and ask it to extract the `citationCount`.
4.  We display the result in the Streamlit app.

**What this proves**: This is our "ground truth" test. It demonstrates the accuracy and reliability of using modern, low-cost LLMs for automated knowledge graph property population.

### Act 2: Multi-Constraint Filtering - The Limits of "Classic RAG"

**Goal**: Show a clear business case where "Classic RAG" fails and the Hybrid Graph approach succeeds.

**Scenario**: "Find active grants in Alabama for amounts less than $500,000."

*   **Column 1: Classic RAG (Vector-Only Search)**
    *   **Mechanism**: We perform a vector search in Pinecone for "grants in Alabama".
    *   **Result**: The search returns semantically similar results, but they are **polluted** with incorrect data (e.g., grants from California, grants for $1,000,000).
    *   **Key Takeaway**: Classic RAG is good for semantic "aboutness" but fails at applying hard, structured filters.

*   **Column 2: Hybrid Graph RAG**
    *   **Mechanism**: We take the IDs from the vector search and apply a simple Cypher query in Neo4j: `... WHERE g.amount <= 500000 AND s.name = 'AL'`.
    *   **Result**: The list is perfectly filtered, showing only the relevant grants.
    *   **Key Takeaway**: The graph adds a layer of precision and reliability that is essential for many business queries.

### Act 3: Contextual Ranking - Beyond Simple Similarity

**Goal**: Demonstrate how the graph allows for more intelligent ranking than pure vector similarity.

**Scenario**: "Find the most influential research on 'graph neural networks'."

*   **Column 1: Classic RAG (Vector-Only Search)**
    *   **Mechanism**: A standard vector search.
    *   **Result**: The top result might be a recent, obscure blog post or a tangentially related paper that happens to match keywords well. The ranking is based only on semantic similarity.
    *   **Key Takeaway**: The "best" result is not always the most semantically similar.

*   **Column 2: Hybrid Graph RAG**
    *   **Mechanism**: We re-rank the vector search results using a hybrid score that combines vector similarity with the paper's `citationCount` from the graph: `hybrid_score = vector_score * log(citationCount + 1)`.
    *   **Result**: A well-known, highly-cited paper rises to the top.
    *   **Key Takeaway**: The graph allows us to blend semantic relevance with domain-specific business logic (like popularity or importance) for superior ranking.

### Act 4: Relationship Discovery - The Unfair Advantage of Graphs

**Goal**: Showcase a query that is fundamentally impossible for a "Classic RAG" system to answer.

**Scenario**: "I'm reading this foundational paper. What other papers should I read that are cited alongside it?" (Co-citation analysis)

*   **Classic RAG**:
    *   **Result**: This question cannot be answered. A vector database has no concept of a "CITES" relationship between document chunks. An attempt to answer would require expensive and unreliable LLM calls over a huge number of documents.

*   **Hybrid Graph RAG**:
    *   **Mechanism**: We execute a Cypher query that traverses the graph: `MATCH (p1:Paper {title: $title})<-[:CITES]-(source)-[:CITES]->(p2:Paper) WHERE p1 <> p2 ...`
    *   **Result**: We get a list of papers that are co-cited, revealing hidden thematic connections.
    *   **Key Takeaway**: This is the unique superpower of a graph database. It can answer questions about complex relationships and network structures that are invisible to other systems. 