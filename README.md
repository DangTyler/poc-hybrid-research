## Hybrid Research PoC (Vector + Graph)

Streamlit demo combining vector search (Pinecone) with graph queries (Neo4j). Primarily tailored for research papers; the grants example demonstrates basic graph filtering.

### Technicalities
- LLM builder (optional, PoC): `LLMGraphTransformer` can infer nodes/edges from abstracts; not production, slower; disabled by default.
- Query: Vector search in Pinecone (OpenAI embeddings) → pass IDs to Neo4j → apply graph filtering and/or re‑ranking.
- Ingestion: Fetch from Semantic Scholar → MERGE `Paper` nodes and `CITES` edges in Neo4j → optional `HAS_TOPIC` → embed to Pinecone.

### How to run or set up

#### Option 1: Hosted Demo
- [Deployed Demo](https://dangtyler-poc-hybrid-research-appdemo-otsorn.streamlit.app/)

#### Option 2: Docker (Recommended for Demos)
```bash
# 1. Create .env file with your credentials (see format below)
# 2. Run quick-start script:

# Linux/Mac
./quick-start.sh

# Windows
quick-start.bat

# Or manually:
docker-compose up --build
```
Access at: http://localhost:8501

#### Option 3: Local Python (Development)
```bash
pip install -r requirements.txt
streamlit run app/demo.py
```
Requires `.env` file with your credentials:
```
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Troubleshooting
- **Database connection error**: Ensure Neo4j Aura instance is running and credentials are correct
- **Import errors**: Run `pip install -r requirements.txt` (or use Docker to avoid dependency issues)
- **No data found**: Run ingestion scripts to populate databases (see ingestion commands below)

### Current demo state
- Focused on research papers; grants are minimal
- No graph analytics (PageRank/community) yet; only live graph filtering and `citationCount` re‑ranking
- Shows generated Cypher; supports small dataset uploads

### Technical challenges
- Cypher quality and safety (schema‑aware prompting, validation)
- Domain modeling/deduplication (IDs, provenance)
- Graph analytics for ranking (future)
- Ingestion performance (batch Neo4j/embeddings/Pinecone)
- LLM limits (don’t use LLMs for citation counts; use authoritative APIs)

### Deploy on Streamlit Community Cloud
1) Push this repo to GitHub
2) New app → Main file `app/demo.py` (Python 3.11)
3) Add Secrets:
```
OPENAI_API_KEY=...
NEO4J_URI=bolt+s://<your-auradb-host>:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
PINECONE_API_KEY=...
```
4) Ensure services are ready:
- Neo4j has your nodes/edges
- Pinecone index `hybrid-research-poc` exists and is populated

### Upload flow (from the UI)
- Choose a dataset name (e.g., `demo_upload`)
- Upload small `.pdf`, `.txt`, or `.md` files
- Optional toggles: LLM Graph Builder, Pinecone embedding
- Ingest tags nodes/edges with the dataset; quality varies with document parsing

### Notes
- Keep secrets out of source; use `.env` locally and Streamlit Cloud Secrets
- Embeddings: `text-embedding-3-small` (1536-dim); Pinecone index name `hybrid-research-poc`
- The app focuses on research papers; grants are a minimal filtering example

### Why domain-specific tuning matters
- Precision/recall: Domain ontologies and normalized schemas reduce ambiguity
- Query reliability: Category-specific Cypher templates and constraints
- Performance: Purposeful projections, indexes, and precomputed properties
- Evaluation: Gold queries and metrics differ by category

Examples
- Research papers: authoritative citation data; `(Paper)-[:CITES]->(Paper)`, `(Paper)-[:WRITTEN_BY]->(Author)`; queries like “Top surveys on X since 2020 with >N citations”
- Grants: `(Grant)-[:ELIGIBLE_FOR]->(State)`, `(Grant)-[:HAS_DEADLINE]->(Deadline)`, `(Grant)-[:FUNDED_BY]->(Agency)`; filters by state, amount caps, deadlines


