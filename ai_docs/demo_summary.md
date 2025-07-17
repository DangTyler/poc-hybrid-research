# PoC Demo - Key Insights & Talking Points

This document summarizes the key insights and technical hurdles overcome during the development of the Hybrid Pinecone + Neo4j PoC.

---

## 1. Research Pipeline (Semantic Scholar -> Neo4j -> Pinecone)

We have successfully built a robust pipeline that fetches research papers, builds a knowledge graph of citations, and creates a searchable vector index.

### Key Insights & Successes:

*   **Hybrid Data Model:** Successfully linked unstructured text search with a structured graph. We store embeddings in Pinecone with a `graph_id` metadata field (`"Paper:<id>"`). This is the core of the hybrid search, allowing us to first find semantically similar papers with Pinecone, then use the `graph_id` to run precise graph queries in Neo4j (e.g., ranking by `citationCount`).
*   **Resilient Data Fetching:** The initial script faced `429 Too Many Requests` errors from the Semantic Scholar API. We implemented a retry mechanism with exponential backoff, demonstrating robust, real-world engineering practices for handling API rate limits.
*   **Data Cleaning & Safeguarding:** The fetched API data had inconsistencies (e.g., `None` values for abstracts or references). The script was enhanced to handle these missing values gracefully, preventing `TypeError` and `IndexError` exceptions and ensuring data quality.

### Technical Hurdles & Solutions:

*   **Environment & Configuration:** A significant portion of the effort was dedicated to debugging environment and configuration issues.
    *   **PowerShell Execution Policy:** Overcame script execution restrictions in the Windows terminal.
    *   **`.env` Loading:** Debugged a persistent issue where credentials were not being loaded. This was traced from a generic "credentials not found" message down to a specific naming mismatch in the `.env` file (`NEO4J_USER` vs. `NEO4J_USERNAME`). This is a great, realistic example of a common development hurdle.
*   **Evolving SDKs:** We encountered and resolved multiple issues related to breaking changes in the Pinecone Python SDK.
    *   **Package Rename:** The `pinecone-client` package was renamed to `pinecone`.
    *   **API Syntax Change:** The initialization logic changed from a top-level `init()` function to a `Pinecone` class instance.
    *   **API Error Handling:** The new SDK provided much clearer error messages, which allowed us to quickly diagnose and fix a problem where the chosen cloud region (`us-west-2`) was not available on the free tier. This demonstrates adaptability to evolving tools and leveraging modern, explicit error handling.

---
*This document will be updated as we progress through the Grants Pipeline and App Development.* 